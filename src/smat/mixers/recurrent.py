#!/usr/bin/env python3
"""The recurrent baselines for the routing benchmark, in chunked form.

    python recurrent.py                 # self-test against the sequential rule

All four are the *recurrences*, wired into the same one-layer ``Router`` as the
other arms (same Q, K, V, O projections, same readout), not the library blocks:
``mamba_ssm`` and ``fla`` are CUDA/Triton-only, and ``fla`` will not even import
under this environment's Python.  What the benchmark needs from them is their
state structure, which is exactly what is reproduced here.

  mamba2    scalar-decay linear recurrence (Mamba-2 / SSD, and GLA with a scalar
            gate):   S_t = a_t S_{t-1} + dt_t k_t v_t^T,   y_t = q_t^T S_t
            with a_t = exp(-dt_t A), dt_t = softplus(w x_t + b) -- the Mamba-2
            parameterisation, initialised as in the reference implementation.

  deltanet  the delta rule (DeltaNet):
            S_t = S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T,  y_t = q_t^T S_t
            with k L2-normalised and beta_t = sigmoid(w x_t) in (0, 1).

  gated_deltanet
            Gated DeltaNet: the delta rule with Mamba-2's scalar decay on the
            retained part,
            S_t = a_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,
            which is ``deltanet`` at a_t = 1 and ``mamba2`` (without dt) at
            beta_t k_t k_t^T = 0.  a_t uses the Mamba-2 parameterisation and
            beta_t the DeltaNet one, which is how the two are combined in the
            reference implementation.

  loglinear log-linear attention: linear attention under a *hierarchical* mask.
            The prefix [1, t] is cut into the O(log t) dyadic blocks of its
            Fenwick decomposition and each block carries its own scalar weight,
            so M_tj = lambda_t^(level(t,j)) and
            o_t = sum_{j <= t} lambda_t^(level(t,j)) (q_t . k_j) v_j.
            lambda_t = softplus(W x_t) in R^H is the input-dependent projection,
            H = ceil(log2 T) + 1.

The first three carry one state of d_qk x d_v; log-linear attention carries
H = Theta(log T) of them.  All four have a row support that is the full causal
prefix -- softplus is strictly positive, so no block weight can vanish -- hence
VC 1, the same as ``M^(1)``.  That is the point of running them: the ceiling at
k = 1 is a property of the support, not of how much the state can hold.

``loglinear`` is evaluated densely, as ``P = A .* M`` with A = Q K^T and M the
hierarchical mask, rather than through the O(T log T) chunked scan.  The two
compute the same function; the scan is an algorithm for it, and at T = 1024 the
dense form is both exact and fast enough.  What this benchmark measures is the
support of M, which the choice of algorithm cannot change.

The chunked forms below are exact, not approximations.  For the delta rule the
within-chunk solve is the UT transform: writing u_j for the j-th update
``beta_j (v_j - S_{j-1}^T k_j)``, the definition gives

    U = diag(b) (V - K S_0) - diag(b) strict_tril(K K^T) U,

one triangular solve per chunk, after which ``S_end = S_0 + K^T U`` and
``Y = Q S_0 + tril(Q K^T) U``.  ``--selftest`` checks both against the loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# sequential references (correctness only -- O(T) python steps)
# ---------------------------------------------------------------------------

def mamba2_loop(Q, K, V, logA, dt):
    nb, T, dk = Q.shape
    dv = V.shape[-1]
    S = Q.new_zeros(nb, dk, dv)
    out = []
    for t in range(T):
        S = logA[:, t, None, None].exp() * S + dt[:, t, None, None] * (
            K[:, t].unsqueeze(-1) @ V[:, t].unsqueeze(-2))
        out.append((Q[:, t].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(out, 1)


def delta_loop(Q, K, V, beta):
    nb, T, dk = Q.shape
    dv = V.shape[-1]
    K = F.normalize(K, dim=-1)
    S = Q.new_zeros(nb, dk, dv)
    out = []
    for t in range(T):
        k, v, b = K[:, t], V[:, t], beta[:, t, None]
        delta = b * (v - (k.unsqueeze(-2) @ S).squeeze(-2))
        S = S + k.unsqueeze(-1) @ delta.unsqueeze(-2)
        out.append((Q[:, t].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(out, 1)


def gated_delta_loop(Q, K, V, logA, beta):
    nb, T, dk = Q.shape
    dv = V.shape[-1]
    K = F.normalize(K, dim=-1)
    S = Q.new_zeros(nb, dk, dv)
    out = []
    for t in range(T):
        k, v, b = K[:, t], V[:, t], beta[:, t, None]
        a = logA[:, t, None, None].exp()
        # a (I - b k k^T) S + b k v^T, written through the delta residual
        keep = a * S
        delta = b * (v - (k.unsqueeze(-2) @ keep).squeeze(-2))
        S = keep + k.unsqueeze(-1) @ delta.unsqueeze(-2)
        out.append((Q[:, t].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(out, 1)


def loglinear_loop(Q, K, V, lam, lev):
    """o_t = sum_{j<=t} lambda_t^(lev[t,j]) (q_t . k_j) v_j, one row at a time."""
    nb, T, _ = Q.shape
    out = []
    for t in range(T):
        w = lam[:, t].gather(1, lev[t, :t + 1].expand(nb, t + 1))     # (nb, t+1)
        a = (Q[:, t].unsqueeze(-2) @ K[:, :t + 1].transpose(-1, -2)).squeeze(-2)
        out.append(((a * w).unsqueeze(-2) @ V[:, :t + 1]).squeeze(-2))
    return torch.stack(out, 1)


# ---------------------------------------------------------------------------
# chunked forms
# ---------------------------------------------------------------------------

def _chunks(x, c):
    nb, T = x.shape[0], x.shape[1]
    return x.reshape(nb, T // c, c, *x.shape[2:])


def mamba2_chunked(Q, K, V, logA, dt, chunk: int = 128):
    nb, T, dk = Q.shape
    dv = V.shape[-1]
    c = min(chunk, T)
    assert T % c == 0, "pad T to a multiple of the chunk before calling"
    nc = T // c
    Qc, Kc, Vc = _chunks(Q, c), _chunks(K, c), _chunks(V, c)
    la = _chunks(logA.unsqueeze(-1), c).squeeze(-1)              # (nb, nc, c)
    dtc = _chunks(dt.unsqueeze(-1), c).squeeze(-1)
    lam = torch.cumsum(la, dim=-1)                               # within chunk
    tri = torch.ones(c, c, device=Q.device).tril()               # j <= i

    # intra-chunk: A_ij = (q_i . k_j) dt_j exp(lam_i - lam_j), j <= i
    W = torch.exp((lam.unsqueeze(-1) - lam.unsqueeze(-2)).masked_fill(
        tri == 0, float("-inf")))                                # (nb,nc,c,c)
    A = (Qc @ Kc.transpose(-1, -2)) * W * dtc.unsqueeze(-2)
    Y = A @ Vc

    # chunk summaries, decayed to the chunk end, and the carry
    wj = torch.exp(lam[..., -1:] - lam) * dtc                    # (nb,nc,c)
    D = (Kc * wj.unsqueeze(-1)).transpose(-1, -2) @ Vc           # (nb,nc,dk,dv)
    aC = torch.exp(lam[..., -1])                                 # (nb,nc)
    S = Q.new_zeros(nb, dk, dv)
    outs = []
    for b in range(nc):
        outs.append(Y[:, b] + (Qc[:, b] * torch.exp(lam[:, b]).unsqueeze(-1)) @ S)
        S = aC[:, b].view(nb, 1, 1) * S + D[:, b]
    return torch.stack(outs, 1).reshape(nb, T, dv)


def delta_chunked(Q, K, V, beta, chunk: int = 128):
    nb, T, dk = Q.shape
    dv = V.shape[-1]
    c = min(chunk, T)
    assert T % c == 0, "pad T to a multiple of the chunk before calling"
    nc = T // c
    K = F.normalize(K, dim=-1)
    Qc, Kc, Vc = _chunks(Q, c), _chunks(K, c), _chunks(V, c)
    bc = _chunks(beta.unsqueeze(-1), c).squeeze(-1)              # (nb, nc, c)

    eye = torch.eye(c, device=Q.device).expand(nb, nc, c, c)
    KK = Kc @ Kc.transpose(-1, -2)
    strict = KK.tril(-1)
    M = eye + bc.unsqueeze(-1) * strict                          # unit lower tri
    QK = (Qc @ Kc.transpose(-1, -2)).tril()                      # j <= i

    S = Q.new_zeros(nb, dk, dv)
    outs = []
    for b in range(nc):
        rhs = bc[:, b].unsqueeze(-1) * (Vc[:, b] - Kc[:, b] @ S)
        U = torch.linalg.solve_triangular(M[:, b], rhs, upper=False,
                                          unitriangular=True)
        outs.append(Qc[:, b] @ S + QK[:, b] @ U)
        S = S + Kc[:, b].transpose(-1, -2) @ U
    return torch.stack(outs, 1).reshape(nb, T, dv)


def gated_delta_chunked(Q, K, V, logA, beta, chunk: int = 128):
    """Gated DeltaNet, exactly, one triangular solve per chunk.

    Writing a_i for the decay and w_i for the update actually written at step i,
    the recurrence is S_i = a_i S_{i-1} + k_i w_i^T with
    w_i = beta_i (v_i - a_i S_{i-1}^T k_i).  Let lam_i = sum_{j<=i} log a_j
    inside the chunk.  Unrolling gives a_i S_{i-1} = e^{lam_i} S_0 +
    sum_{j<i} e^{lam_i - lam_j} k_j w_j^T, so

        (I + P) W = diag(beta) (V - diag(e^lam) K S_0),
        P_ij = beta_i e^{lam_i - lam_j} (k_i . k_j)   for j < i,

    one unit-lower-triangular solve, after which

        Y = diag(e^lam) Q S_0 + G W,   G_ij = e^{lam_i - lam_j} (q_i . k_j), j<=i
        S_end = e^{lam_c} S_0 + K^T diag(e^{lam_c - lam}) W.

    At logA = 0 every exponential is 1 and this is ``delta_chunked`` term for
    term; at beta = 0 it is the scalar-decay recurrence.  ``_selftest`` checks
    both limits as well as the general case.
    """
    nb, T, dk = Q.shape
    dv = V.shape[-1]
    c = min(chunk, T)
    assert T % c == 0, "pad T to a multiple of the chunk before calling"
    nc = T // c
    K = F.normalize(K, dim=-1)
    Qc, Kc, Vc = _chunks(Q, c), _chunks(K, c), _chunks(V, c)
    bc = _chunks(beta.unsqueeze(-1), c).squeeze(-1)              # (nb, nc, c)
    la = _chunks(logA.unsqueeze(-1), c).squeeze(-1)
    lam = torch.cumsum(la, dim=-1)                               # (nb, nc, c)

    tri = torch.ones(c, c, device=Q.device).tril()               # j <= i
    # e^{lam_i - lam_j} on j <= i, zero above the diagonal; lam is decreasing,
    # so every entry is in (0, 1] and no exponential can overflow.
    Wd = torch.exp((lam.unsqueeze(-1) - lam.unsqueeze(-2)).masked_fill(
        tri == 0, float("-inf")))                                # (nb,nc,c,c)
    KK = (Kc @ Kc.transpose(-1, -2)) * Wd
    eye = torch.eye(c, device=Q.device).expand(nb, nc, c, c)
    M = eye + bc.unsqueeze(-1) * KK.tril(-1)                     # unit lower tri
    G = (Qc @ Kc.transpose(-1, -2)) * Wd                         # already masked
    el = torch.exp(lam)                                          # (nb,nc,c)

    S = Q.new_zeros(nb, dk, dv)
    outs = []
    for b in range(nc):
        rhs = bc[:, b].unsqueeze(-1) * (
            Vc[:, b] - el[:, b].unsqueeze(-1) * (Kc[:, b] @ S))
        W = torch.linalg.solve_triangular(M[:, b], rhs, upper=False,
                                          unitriangular=True)
        outs.append(el[:, b].unsqueeze(-1) * (Qc[:, b] @ S) + G[:, b] @ W)
        wj = torch.exp(lam[:, b, -1:] - lam[:, b])               # (nb, c)
        S = el[:, b, -1].view(nb, 1, 1) * S + \
            (Kc[:, b] * wj.unsqueeze(-1)).transpose(-1, -2) @ W
    return torch.stack(outs, 1).reshape(nb, T, dv)


def fenwick_levels(T: int) -> torch.Tensor:
    """``lev[t, j]`` = which dyadic block of the Fenwick cut of [1, t] holds j.

    The prefix is cut by stripping the lowest set bit: [1, t] becomes the block
    of size lowbit(t) ending at t, then the same on t - lowbit(t).  A block of
    size 2^l is at level l, so row t names at most popcount(t) <= ceil(log2 T)
    distinct levels and every j <= t gets exactly one.  Positions above the
    diagonal are marked -1 and are never read.
    """
    lev = torch.full((T, T), -1, dtype=torch.long)
    for t in range(1, T + 1):
        x = t
        while x > 0:
            lb = x & (-x)
            lev[t - 1, x - lb:x] = lb.bit_length() - 1
            x -= lb
    return lev


def loglinear_attention(Q, K, V, lam, lev, chunk: int = 128):
    """Linear attention under the hierarchical mask, in query blocks.

    ``lam`` is (nb, T, H) and strictly positive, ``lev`` is the (T, T) table
    from ``fenwick_levels``.  Blocking over queries keeps the (nb, c, T) score
    and weight matrices small; the result is the dense product, not an
    approximation of it.
    """
    nb, T, _ = Q.shape
    c = min(chunk, T)
    idx = torch.arange(T, device=Q.device)
    outs = []
    for s in range(0, T, c):
        e = min(s + c, T)
        L = lev[s:e]                                             # (c, T)
        keep = (idx.unsqueeze(0) <= idx[s:e].unsqueeze(1))       # (c, T) causal
        rows = torch.arange(s, e, device=Q.device).unsqueeze(1).expand_as(L)
        M = lam[:, rows, L.clamp_min(0)] * keep                  # (nb, c, T)
        A = Q[:, s:e] @ K.transpose(-1, -2)
        outs.append((A * M) @ V)
    return torch.cat(outs, 1)


# ---------------------------------------------------------------------------
# the parameters each arm adds to the Router
# ---------------------------------------------------------------------------

class Mamba2Gate(nn.Module):
    """dt and A of the Mamba-2 parameterisation, one head."""

    def __init__(self, d_model: int, dt_init=(1e-3, 1e-1), A_init=(1.0, 16.0)):
        super().__init__()
        self.w_dt = nn.Linear(d_model, 1)
        lo, hi = dt_init
        u = torch.rand(1) * (torch.log(torch.tensor(hi)) - torch.log(torch.tensor(lo)))
        dt = torch.exp(torch.log(torch.tensor(lo)) + u)
        with torch.no_grad():
            self.w_dt.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(1).uniform_(*A_init)))

    def forward(self, x):
        dt = F.softplus(self.w_dt(x)).squeeze(-1)                # (nb, T) > 0
        return -dt * self.A_log.exp(), dt                        # logA <= 0, dt


class DeltaGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.w_b = nn.Linear(d_model, 1)

    def forward(self, x):
        return torch.sigmoid(self.w_b(x)).squeeze(-1)            # (nb, T) in (0,1)


class GatedDeltaGate(nn.Module):
    """Gated DeltaNet's two gates: Mamba-2's decay and DeltaNet's step size.

    Kept as the two reference parameterisations rather than a new one, so the
    arm is the combination of the two baselines beside it and nothing else.
    """

    def __init__(self, d_model: int, dt_init=(1e-3, 1e-1), A_init=(1.0, 16.0)):
        super().__init__()
        self.decay = Mamba2Gate(d_model, dt_init=dt_init, A_init=A_init)
        self.w_b = nn.Linear(d_model, 1)

    def forward(self, x):
        logA, _ = self.decay(x)                                  # dt scales the
        beta = torch.sigmoid(self.w_b(x)).squeeze(-1)            # write instead
        return logA, beta                                        # logA <= 0


class LogLinearGate(nn.Module):
    """lambda_t in R^H, strictly positive, from an input-dependent projection.

    softplus rather than anything that can reach zero: a vanishing block weight
    would puncture the row support, and the claim being tested is about a mask
    whose support is the whole prefix.  Initialised small so the layer starts
    near a uniform hierarchical mask.
    """

    def __init__(self, d_model: int, T: int):
        super().__init__()
        self.H = T.bit_length()                  # ceil(log2 T) + 1 for T = 2^m
        self.w = nn.Linear(d_model, self.H)
        nn.init.normal_(self.w.weight, std=0.02)
        nn.init.zeros_(self.w.bias)

    def forward(self, x):
        return F.softplus(self.w(x))                             # (nb, T, H) > 0


# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    nb, T, dk, dv = 3, 64, 8, 6
    Q, K, V = (torch.randn(nb, T, dk), torch.randn(nb, T, dk),
               torch.randn(nb, T, dv))
    dt = torch.rand(nb, T) * 0.1 + 1e-3
    logA = -dt * 2.0
    beta = torch.rand(nb, T)
    lev = fenwick_levels(T)
    lam = F.softplus(torch.randn(nb, T, T.bit_length()))
    ok = True
    for c in (8, 16, 64):
        e1 = (mamba2_chunked(Q, K, V, logA, dt, c) - mamba2_loop(Q, K, V, logA, dt)).abs().max()
        e2 = (delta_chunked(Q, K, V, beta, c) - delta_loop(Q, K, V, beta)).abs().max()
        e3 = (gated_delta_chunked(Q, K, V, logA, beta, c)
              - gated_delta_loop(Q, K, V, logA, beta)).abs().max()
        e4 = (loglinear_attention(Q, K, V, lam, lev, c)
              - loglinear_loop(Q, K, V, lam, lev)).abs().max()
        print(f"  chunk {c:3d}:  mamba2 {e1:.2e}   deltanet {e2:.2e}"
              f"   gated_deltanet {e3:.2e}   loglinear {e4:.2e}")
        ok &= bool(e1 < 1e-4 and e2 < 1e-4 and e3 < 1e-4 and e4 < 1e-4)

    # the two limits that say the gated arm really is the combination it claims
    z = torch.zeros(nb, T)
    e5 = (gated_delta_chunked(Q, K, V, z, beta, 16)
          - delta_chunked(Q, K, V, beta, 16)).abs().max()
    e6 = (gated_delta_chunked(Q, K, V, logA, z, 16)).abs().max()
    print(f"  limits:     logA=0 vs deltanet {e5:.2e}   beta=0 gives {e6:.2e}")
    ok &= bool(e5 < 1e-5 and e6 < 1e-5)

    # the hierarchical mask covers the causal triangle exactly: every j <= t is
    # in one block and nothing above the diagonal is.  This is the support
    # claim the loglinear arm is in the benchmark to test.
    i = torch.arange(T)
    causal = i.unsqueeze(0) <= i.unsqueeze(1)                    # j <= t
    cover = bool(((lev >= 0) == causal).all())
    per_row = int((lev >= 0).sum(1).max()), int(lev.max().item()) + 1
    print(f"  fenwick:    support == causal triangle: {cover}"
          f"   levels {per_row[1]} = ceil(log2 {T}) + 1")
    ok &= cover and per_row[1] == T.bit_length()

    print("chunked forms match the sequential rule" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

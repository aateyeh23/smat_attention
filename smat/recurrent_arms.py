#!/usr/bin/env python3
"""Two recurrent baselines for the routing benchmark, in chunked form.

    python recurrent_arms.py            # self-test against the sequential rule

Both are the *recurrences*, wired into the same one-layer ``Router`` as the
other arms (same Q, K, V, O projections, same readout), not the library blocks:
``mamba_ssm`` and ``fla`` are CUDA/Triton-only and this sweep runs on CPU.  What
the benchmark needs from them is their state structure, which is exactly what is
reproduced here.

  mamba2    scalar-decay linear recurrence (Mamba-2 / SSD, and GLA with a scalar
            gate):   S_t = a_t S_{t-1} + dt_t k_t v_t^T,   y_t = q_t^T S_t
            with a_t = exp(-dt_t A), dt_t = softplus(w x_t + b) -- the Mamba-2
            parameterisation, initialised as in the reference implementation.

  deltanet  the delta rule (DeltaNet):
            S_t = S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T,  y_t = q_t^T S_t
            with k L2-normalised and beta_t = sigmoid(w x_t) in (0, 1).

Both carry one state of d_qk x d_v, hence a row support that is the full causal
prefix: VC 1, the same as ``M^(1)``.  That is the point of running them.

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


# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    nb, T, dk, dv = 3, 64, 8, 6
    Q, K, V = (torch.randn(nb, T, dk), torch.randn(nb, T, dk),
               torch.randn(nb, T, dv))
    dt = torch.rand(nb, T) * 0.1 + 1e-3
    logA = -dt * 2.0
    beta = torch.rand(nb, T)
    ok = True
    for c in (8, 16, 64):
        e1 = (mamba2_chunked(Q, K, V, logA, dt, c) - mamba2_loop(Q, K, V, logA, dt)).abs().max()
        e2 = (delta_chunked(Q, K, V, beta, c) - delta_loop(Q, K, V, beta)).abs().max()
        print(f"  chunk {c:3d}:  mamba2 {e1:.2e}   deltanet {e2:.2e}")
        ok &= bool(e1 < 1e-4 and e2 < 1e-4)
    print("chunked forms match the sequential rule" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

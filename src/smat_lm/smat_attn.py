#!/usr/bin/env python3
"""SMAT attention under the revised Section 3.3 schedule.

Everything operates on kernel features

    Phi_i = phi_Q(q_i),   Psi_j = h_j phi_K(k_j),   Vb_j = [v_j, 1] in R^p,

so a single code path carries the numerator and the normalizer of

    o_i = ( Phi_i^T sum_j M_ij Psi_j Vb_j^T )_{1:d_v}  /  ( ... )_p .

The mask factorizes (Eq. 3.13) as

    M^(d) = [ L_n   0     ]  +  [ 0        0 ]
            [ 0     L_{TR} ]     [ R C S^T  0 ]

so the schedule is exactly two passes, summed *before* the single division:

  causal      one chunked segmented scan over all T with one reset at n;
  long range  pool by profile (S^T), contract with the incidence (C), then one
              type-major GEMM per row type (R).

The three long-range factors are applied in that order and never form G.  The
object whose sparsity governs cost is C, with nnz(C) = Theta(T^{2-3/d}).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Dict

import numpy as np
import torch

from smat_mask import MaskSpec

try:
    import smat_triton as _tri
except Exception:                                       # pragma: no cover
    _tri = None

__all__ = [
    "make_phi", "featurise", "to_device",
    "segmented_causal_scan", "pool_profiles", "apply_incidence",
    "query_by_type", "zeta_transform", "walsh_incidence",
    "smat_attention", "dense_masked_kernel_attention", "SmatDecoder",
    "have_triton",
]


_TRITON_DEAD: set = set()


def have_triton() -> bool:
    return _tri is not None and _tri.available()


def _try_triton(fn, *a, **kw):
    """Run a Triton kernel; on failure retire *that* kernel for the session.

    A long unattended sweep should degrade to the torch path with a loud warning
    rather than abort, but one kernel failing is no reason to give up the
    others.  An explicit ``backend="triton"`` still raises, so the tests can
    tell a fallback from a pass.
    """
    if fn.__name__ in _TRITON_DEAD:
        return None
    try:
        return fn(*a, **kw)
    except Exception as e:                                  # pragma: no cover
        _TRITON_DEAD.add(fn.__name__)
        print(f"WARNING: triton kernel {fn.__name__} failed ({type(e).__name__}: "
              f"{e}); using the torch path for that step for the rest of this "
              f"run.  Other kernels are unaffected.")
        return None


# ---------------------------------------------------------------------------
# feature map
# ---------------------------------------------------------------------------

def make_phi(d_qk: int, r: int, *, kind: str = "elu1", device="cpu",
             dtype=torch.float32, seed: int = 0) -> Callable:
    """A strictly positive finite-rank feature map ``R^{d_qk} -> R^r_{>=0}``.

    Eq. (3.14) is a genuine hypothesis: softmax is not of this form.  The map
    must be nonnegative so the normalizer stays a sum of nonnegative terms.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    W = (torch.randn(r, d_qk, generator=gen, dtype=torch.float64)
         / np.sqrt(d_qk)).to(device=device, dtype=dtype)
    if kind == "elu1":
        return lambda x: torch.nn.functional.elu(x @ W.T) + 1.0
    if kind == "relu":
        return lambda x: torch.nn.functional.relu(x @ W.T) + 1e-3
    raise ValueError(f"unknown feature map {kind!r}")


def featurise(Q, K, V, phi, *, g=None, h=None):
    """``(Q,K,V) -> (Phi, Psi, Vb)`` with the key gate folded into ``Psi``."""
    Phi = phi(Q)
    Psi = phi(K)
    if h is not None:
        Psi = Psi * h.unsqueeze(-1)
    ones = torch.ones(*V.shape[:-1], 1, device=V.device, dtype=V.dtype)
    return Phi, Psi, torch.cat([V, ones], dim=-1)


_TRI_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _tri_ok(x, acc) -> bool:
    """Whether the fused kernels may be used for this tensor and accumulator.

    They accumulate and return fp32, so the fp64 reference path -- which is what
    every correctness and conditioning measurement is compared against -- must
    stay on torch.  Silently downgrading it would make the reference agree with
    whatever it was meant to check.
    """
    return (x.is_cuda and x.dtype in _TRI_DTYPES
            and acc in (None, torch.float32))


def _acc_of(x, acc_dtype=None):
    if acc_dtype is not None:
        return acc_dtype
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype


# ---------------------------------------------------------------------------
# causal pass: one segmented chunked scan, one reset at n
# ---------------------------------------------------------------------------

def segmented_causal_scan(Phi, Psi, Vb, n: int, chunk: int = 128,
                          acc_dtype=None, backend: str = "auto"):
    """``[[L_n, 0], [0, L_{T_R}]] Phi Psi^T Vb`` as a single scan with one reset.

    ``n`` must be a multiple of ``chunk`` so the segment boundary falls on a
    chunk boundary; :func:`smat_mask.build_mask` guarantees this.  The only
    inter-chunk communication is one ``r x p`` state.
    """
    nb, T, r = Phi.shape
    p = Vb.shape[-1]
    C = min(chunk, T)
    if n % C:
        raise ValueError(f"segment boundary n={n} is not a multiple of chunk={C}")
    acc = _acc_of(Phi, acc_dtype)

    pad = (-T) % C
    if pad:                                  # the tail lives in the recent segment
        Phi = torch.nn.functional.pad(Phi, (0, 0, 0, pad))
        Psi = torch.nn.functional.pad(Psi, (0, 0, 0, pad))
        Vb = torch.nn.functional.pad(Vb, (0, 0, 0, pad))
    nc = (T + pad) // C
    nc0 = n // C                             # chunks in the landmark segment

    Phi_c = Phi.reshape(nb, nc, C, r)
    Psi_c = Psi.reshape(nb, nc, C, r)
    Vb_c = Vb.reshape(nb, nc, C, p)

    # per-chunk summary D_b, then a segmented exclusive scan
    D = Psi_c.to(acc).transpose(-1, -2) @ Vb_c.to(acc)              # (nb,nc,r,p)
    # The reset is done by scanning each segment on its own, NOT by taking a
    # global cumsum and subtracting the landmark total back out.  The two agree
    # in exact arithmetic, but the subtractive form routes the whole landmark
    # block's mass through every recent chunk's prefix, where it fails to cancel
    # to O(eps): a landmark payload then perturbs a recent row that the mask
    # excludes it from.  That is the cancellation Remark 3.10 rejects for the
    # long-range route, and it has no place in the causal one -- segmenting the
    # scan costs the same and makes M^(d)'s zero entries exactly non-influential.
    Sin = torch.zeros_like(D)
    bounds = [(0, nc0), (nc0, nc)] if 0 < nc0 < nc else [(0, nc)]
    for lo, hi in bounds:
        if hi - lo > 1:
            # A *shifted* scan, not cumsum-minus-self: (a + x) - x is not a in
            # floating point, so the subtractive form lets D_b perturb its own
            # S^in_b and thence rows of chunk b that precede the perturbed
            # column -- rows the mask excludes it from.
            Sin[:, lo + 1:hi] = torch.cumsum(D[:, lo:hi - 1], dim=1)

    out = None
    if backend == "triton":
        out = _tri.chunk_local(Phi_c, Psi_c, Vb_c, Sin)
    elif backend == "auto" and have_triton() and _tri_ok(Phi, acc):
        out = _try_triton(_tri.chunk_local, Phi_c, Psi_c, Vb_c, Sin)
    if out is None:
        inter = Phi_c.to(acc) @ Sin
        tri = torch.ones(C, C, device=Phi.device, dtype=Phi.dtype).tril()
        A = (Phi_c @ Psi_c.transpose(-1, -2)) * tri
        out = inter + A.to(acc) @ Vb_c.to(acc)

    return out.reshape(nb, T + pad, p)[:, :T]


# ---------------------------------------------------------------------------
# long-range pass: S^T, then C, then R
# ---------------------------------------------------------------------------

def pool_profiles(Psi, Vb, spec: MaskSpec, acc_dtype=None,
                  backend: str = "auto"):
    """``F_x = sum_{prof(j)=x} Psi_j Vb_j^T``  --  apply ``S^T``.

    ``prof(j) = (j - n_P) mod N0`` is round-robin, so the columns of a profile
    are a *strided slice*: no gather is needed, and the whole pooling step is one
    batched GEMM of ``N0`` problems of shape ``(r x g)(g x p)``, costing
    ``n_X * r * p`` MACs.
    """
    nb = Psi.shape[0]
    r, p = Psi.shape[-1], Vb.shape[-1]
    acc = _acc_of(Psi, acc_dtype)
    n_P, n, N0 = spec.n_P, spec.n, spec.N0
    gmax = -(-(n - n_P) // N0)

    if backend == "triton" or (backend == "auto" and have_triton()
                               and _tri_ok(Psi, acc)
                               and _tri.pool_ok(N0, gmax, Vb.shape[-1])):
        if backend == "triton":
            return _tri.pool(Psi, Vb, n_P, n, N0)
        F = _try_triton(_tri.pool, Psi, Vb, n_P, n, N0)
        if F is not None:
            return F

    Ps = Psi[:, n_P:n]
    Vs = Vb[:, n_P:n]
    n_X = n - n_P
    g, rem = divmod(n_X, N0)

    F = torch.zeros(nb, N0, r, p, device=Psi.device, dtype=acc)
    if g:
        A = Ps[:, :g * N0].reshape(nb, g, N0, r).permute(0, 2, 3, 1)   # (nb,N0,r,g)
        Bm = Vs[:, :g * N0].reshape(nb, g, N0, p).permute(0, 2, 1, 3)  # (nb,N0,g,p)
        F = A.to(acc) @ Bm.to(acc)
    if rem:
        F[:, :rem] += (Ps[:, g * N0:].to(acc).unsqueeze(-1)
                       * Vs[:, g * N0:].to(acc).unsqueeze(-2))
    return F


def apply_incidence(F, spec: MaskSpec, *, backend: str = "auto",
                    gather_chunk: int = 0):
    """``U_h = sum_{x in H_h} F_x``  --  apply ``C``.

    Cost is ``nnz(C) * r * p`` adds; the intermediate ``(nb, B, deg, r, p)`` is
    never formed.  ``backend`` selects the realisation: ``"triton"`` for the
    fused gather-reduce kernel, ``"torch"`` for a chunked gather, ``"walsh"``
    for the endpoint's fast Walsh-Hadamard route.
    """
    nb, N0, r, p = F.shape
    B = spec.B

    if spec.kind == "endpoint" and backend in ("auto", "walsh"):
        return walsh_incidence(F)

    if backend == "triton":
        if spec.plane_pts_t is None:
            raise ValueError("the triton incidence kernel needs a uniform row "
                             "degree; this mask is ragged (use walsh or torch)")
        return _tri.incidence(F, spec.plane_pts_t, B)
    if backend == "auto":
        if (have_triton() and spec.plane_pts_t is not None
                and F.is_cuda and F.dtype == torch.float32):
            U = _try_triton(_tri.incidence, F, spec.plane_pts_t, B)
            if U is not None:
                return U
        backend = "torch"

    plane = spec.plane_pts_t
    if plane is None:                         # ragged: fall back to CSR loops
        U = torch.zeros(nb, B, r, p, device=F.device, dtype=F.dtype)
        ptr = spec.C_indptr
        for h in range(B):
            idx = spec.C_indices_t[int(ptr[h]):int(ptr[h + 1])]
            U[:, h] = F[:, idx].sum(1)
        return U

    deg = plane.shape[1]
    # budget the gather at ~1 GB of fp32 so the torch path is a fair comparison
    # rather than an allocator-thrash strawman
    step = gather_chunk or max(1, min(deg, (1 << 28) // max(1, nb * B * r * p)))
    U = torch.zeros(nb, B, r, p, device=F.device, dtype=F.dtype)
    for lo in range(0, deg, step):
        idx = plane[:, lo:lo + step].reshape(-1).long()
        s = idx.numel() // B
        U += F[:, idx].reshape(nb, B, s, r, p).sum(2)
    return U


def walsh_incidence(F):
    """``C = (J + H)/2`` with ``H`` the Walsh matrix: apply ``C`` in ``O(B log B)``.

    For the endpoint mask ``C_{ha} = 1{<h,a> = 0} = (1 + (-1)^{<h,a>})/2``, so
    the ``Theta(T^2)`` incidence is applied by a ``k``-stage butterfly instead of
    being enumerated.  Exact in exact arithmetic; the halving is a power of two.
    """
    nb, N, r, p = F.shape
    k = N.bit_length() - 1
    assert 1 << k == N, "Walsh route needs a power-of-two profile count"
    tot = F.sum(1, keepdim=True)
    H = F.reshape(nb, N, r * p).clone()
    for j in range(k):
        lo = 1 << j
        H = H.reshape(nb, N // (2 * lo), 2, lo, r * p)
        a, b = H[:, :, 0].clone(), H[:, :, 1].clone()
        H[:, :, 0] = a + b
        H[:, :, 1] = a - b
        H = H.reshape(nb, N, r * p)
    return 0.5 * (tot + H.reshape(nb, N, r, p))


def query_by_type(Phi_rec, U, spec: MaskSpec):
    """``Ybar^lr_{R_h} = Phi_{R_h} U_h``  --  apply ``R``.

    ``rho(i) = i mod B`` makes ``R_h`` a stride-``B`` slice, so a reshape and a
    transpose line every recent query up with its type and the pass is a single
    batched GEMM.  The tail of ``T_R mod B`` rows is one extra small product.
    """
    nb, T_R, r = Phi_rec.shape
    B, p = spec.B, U.shape[-1]
    acc = U.dtype
    out = torch.empty(nb, T_R, p, device=Phi_rec.device, dtype=acc)
    c = T_R // B
    if c:
        head = Phi_rec[:, :c * B].reshape(nb, c, B, r).transpose(1, 2)   # (nb,B,c,r)
        out[:, :c * B] = ((head.to(acc) @ U).transpose(1, 2)
                          .reshape(nb, c * B, p))
    rem = T_R - c * B
    if rem:
        tail = Phi_rec[:, c * B:].to(acc).unsqueeze(-2)                  # (nb,rem,1,r)
        out[:, c * B:] = (tail @ U[:, :rem]).squeeze(-2)
    return out


def direct_long_range(Psi, Vb, spec: MaskSpec, acc_dtype=None,
                      type_chunk: int = 64):
    """The *unpooled* backend: scatter every landmark column to its hyperplanes.

    Costs ``I_tok * r * p = Theta(T^{2-2/d})`` MACs -- a factor ``~q ~ T^{1/d}``
    above the pooled route.  Kept only as the comparison of Sec. 3.3.
    """
    nb = Psi.shape[0]
    r, p = Psi.shape[-1], Vb.shape[-1]
    acc = _acc_of(Psi, acc_dtype)
    cols = spec.type_cols_t                      # (B, kappa_max), padded with n_X
    if cols is None:
        raise ValueError("the direct backend needs to_device(..., "
                         "want_direct=True)")
    B, kap = cols.shape

    Ps = torch.cat([Psi[:, spec.n_P:spec.n],
                    Psi.new_zeros(nb, 1, r)], dim=1)
    Vs = torch.cat([Vb[:, spec.n_P:spec.n],
                    Vb.new_zeros(nb, 1, p)], dim=1)

    U = torch.empty(nb, B, r, p, device=Psi.device, dtype=acc)
    for lo in range(0, B, type_chunk):
        hi = min(lo + type_chunk, B)
        c = cols[lo:hi].reshape(-1).long()
        nr = hi - lo
        Pk = Ps[:, c].reshape(nb, nr, kap, r)
        Vk = Vs[:, c].reshape(nb, nr, kap, p)
        U[:, lo:hi] = Pk.to(acc).transpose(-1, -2) @ Vk.to(acc)
    return U


# ---------------------------------------------------------------------------
# the endpoint's landmark block: the Boolean zeta transform
# ---------------------------------------------------------------------------

def zeta_transform(F):
    """``(Z_k F)_a = sum_{x subseteq a} F_x`` by a ``k``-stage butterfly.

    ``Z_k = L_2^{otimes k}``, so the subset-sum is ``O(n log n)`` rather than the
    ``O(n^2)`` the dense block would cost.  In-place over the leading axis.
    """
    nb, N, r, p = F.shape
    k = N.bit_length() - 1
    assert 1 << k == N
    H = F.reshape(nb, N, r * p).clone()
    for j in range(k):
        lo = 1 << j
        H = H.reshape(nb, N // (2 * lo), 2, lo, r * p)
        H[:, :, 1] += H[:, :, 0]
        H = H.reshape(nb, N, r * p)
    return H.reshape(nb, N, r, p)


# ---------------------------------------------------------------------------
# full forward
# ---------------------------------------------------------------------------

def smat_attention(Phi, Psi, Vb, spec: MaskSpec, *, chunk: int = 128, g=None,
                   acc_dtype=None, timer=None, long_range: str = "pooled",
                   scan_backend: str = "auto", incidence_backend: str = "auto",
                   return_parts: bool = False):
    """Algorithm 1 of Sec. 3.3.

    ``long_range`` is ``"pooled"`` (the schedule of the draft), ``"direct"``
    (the unpooled comparison), or ``"subtractive"`` (Remark 3.10 -- computed,
    not recommended).  Contributions are summed before the single division.
    """
    tm = timer or (lambda name: nullcontext())
    nb, T, r = Phi.shape
    p = Vb.shape[-1]
    n, T_R = spec.n, spec.T_R
    acc = _acc_of(Phi, acc_dtype)
    parts: Dict[str, torch.Tensor] = {}

    if long_range == "subtractive":
        return _subtractive(Phi, Psi, Vb, spec, chunk=chunk, g=g, acc=acc, tm=tm,
                            return_parts=return_parts,
                            incidence_backend=incidence_backend)

    # ---- long range: S^T, then C, then R ---------------------------------
    with tm("lr_pool"):
        F = pool_profiles(Psi, Vb, spec, acc_dtype=acc,
                          backend=incidence_backend)
    with tm("lr_incidence"):
        if long_range == "direct":
            U = direct_long_range(Psi, Vb, spec, acc_dtype=acc)
        else:
            U = apply_incidence(F, spec, backend=incidence_backend)
        if spec.n_P:                                   # Remark 3.4: rank one
            b = (Psi[:, :spec.n_P].to(acc).transpose(-1, -2)
                 @ Vb[:, :spec.n_P].to(acc))
            U = U + b.unsqueeze(1)
    with tm("lr_query"):
        Y_lr = query_by_type(Phi[:, n:], U, spec)

    # ---- causal ----------------------------------------------------------
    if spec.kind == "endpoint":
        with tm("zeta"):
            Fz = (Psi[:, :n].to(acc).unsqueeze(-1)
                  * Vb[:, :n].to(acc).unsqueeze(-2))
            Zt = zeta_transform(Fz)
            top = (Phi[:, :n].to(acc).unsqueeze(-2) @ Zt).squeeze(-2)
        with tm("scan"):
            loc = segmented_causal_scan(Phi[:, n:], Psi[:, n:], Vb[:, n:],
                                        n=0, chunk=chunk, acc_dtype=acc,
                                        backend=scan_backend)
        H = torch.cat([top, loc + Y_lr], dim=1)
        parts["zeta"] = top
    else:
        with tm("scan"):
            H = segmented_causal_scan(Phi, Psi, Vb, n=n, chunk=chunk,
                                      acc_dtype=acc, backend=scan_backend)
        H = torch.cat([H[:, :n], H[:, n:] + Y_lr], dim=1)

    parts.update(U=U, Y_lr=Y_lr, H=H)

    with tm("normalise"):
        out = H[..., :-1] / H[..., -1:]
        if g is not None:
            out = out * g.unsqueeze(-1).to(out.dtype)

    return (out, parts) if return_parts else out


def _subtractive(Phi, Psi, Vb, spec, *, chunk, g, acc, tm, return_parts,
                 incidence_backend):
    """Remark 3.10: one unbroken causal scan minus the excluded long-range mass.

    ``M^(d) = L_T - R Cbar S^T`` with ``Cbar = J - C``.  ``Ubar_h = F_tot - U_h``
    so the dense complement is never formed and the arithmetic cost is the same.
    Implemented to *measure* the cancellation it introduces into the normalizer,
    which is why the draft does not adopt it.
    """
    n = spec.n
    with tm("lr_pool"):
        F = pool_profiles(Psi, Vb, spec, acc_dtype=acc)
    with tm("lr_incidence"):
        U = apply_incidence(F, spec, backend=incidence_backend)
        Ubar = F.sum(1, keepdim=True) - U                  # (nb,B,r,p)
    with tm("lr_query"):
        Y_ex = query_by_type(Phi[:, n:], Ubar, spec)
    with tm("scan"):
        H = segmented_causal_scan(Phi, Psi, Vb, n=0, chunk=chunk, acc_dtype=acc)
    H = torch.cat([H[:, :n], H[:, n:] - Y_ex], dim=1)
    with tm("normalise"):
        out = H[..., :-1] / H[..., -1:]
        if g is not None:
            out = out * g.unsqueeze(-1).to(out.dtype)
    return (out, {"H": H, "Y_ex": Y_ex}) if return_parts else out


def dense_masked_kernel_attention(Phi, Psi, Vb, M):
    """``O(T^2)`` reference straight from Eq. (3.16).  Verification only."""
    A = torch.einsum("nta,nsa->nts", Phi, Psi) * M
    H = torch.einsum("nts,nsv->ntv", A, Vb)
    return H[..., :-1] / H[..., -1:]


# ---------------------------------------------------------------------------
# streaming decode (Remark 3.9)
# ---------------------------------------------------------------------------

class SmatDecoder:
    """Exact autoregressive decoding at ``O(rp)`` per token, flat in ``T``.

    Every landmark key precedes every recent query, so the type table ``{U_h}``
    is fixed once prefill completes and no auxiliary structure is needed:
    step ``n+i`` is ``Phi^T (U_{rho(i)} + sum_{n<u<=n+i} Z_u)``.
    """

    def __init__(self, Psi_prefix, Vb_prefix, spec: MaskSpec, *, acc_dtype=None,
                 incidence_backend: str = "auto"):
        self.spec = spec
        acc = _acc_of(Psi_prefix, acc_dtype)
        self.acc = acc
        nb, npre, r = Psi_prefix.shape
        p = Vb_prefix.shape[-1]
        self.nb, self.r, self.p = nb, r, p

        F = pool_profiles(Psi_prefix, Vb_prefix, spec, acc_dtype=acc)
        self.U = apply_incidence(F, spec, backend=incidence_backend)
        if spec.n_P:
            b = (Psi_prefix[:, :spec.n_P].to(acc).transpose(-1, -2)
                 @ Vb_prefix[:, :spec.n_P].to(acc))
            self.U = self.U + b.unsqueeze(1)
        self.S = torch.zeros(nb, r, p, device=Psi_prefix.device, dtype=acc)
        self.i = 0

    def step(self, Phi_t, Psi_t, Vb_t):
        """One token.  ``Phi_t, Psi_t: (nb, r)``, ``Vb_t: (nb, p)``."""
        self.S += Psi_t.to(self.acc).unsqueeze(-1) * Vb_t.to(self.acc).unsqueeze(-2)
        H = self.S + self.U[:, self.i % self.spec.B]
        o = (Phi_t.to(self.acc).unsqueeze(-2) @ H).squeeze(-2)
        self.i += 1
        return o[..., :-1] / o[..., -1:]

    def cache_words(self) -> int:
        return int(self.S.numel() + self.U.numel())

    @staticmethod
    def kv_cache_words(nb: int, T: int, d_v: int) -> int:
        return int(nb * 2 * T * d_v)


# ---------------------------------------------------------------------------
# device staging
# ---------------------------------------------------------------------------

def to_device(spec: MaskSpec, device, *, want_direct: bool = False) -> MaskSpec:
    """Stage the incidence (and optionally the direct column table) on ``device``."""
    spec.C_indptr_t = torch.from_numpy(spec.C_indptr).to(device)
    spec.C_indices_t = torch.from_numpy(spec.C_indices.astype(np.int64)).to(device)
    if spec.uniform_deg:
        spec.plane_pts_t = spec.C_indices_t.view(spec.B, spec.uniform_deg)
    else:
        spec.plane_pts_t = None
    if want_direct:
        if not spec.uniform_deg:
            raise ValueError("the direct backend needs a uniform row degree; "
                             "the endpoint mask is ragged")
        spec.type_cols_t = torch.from_numpy(_type_columns(spec)).to(device)
    return spec


def _type_columns(spec: MaskSpec) -> np.ndarray:
    """``(B, kappa)`` landmark columns per row type, padded with ``n_X``.

    Only the direct (unpooled) comparison needs this.  It is ``Theta(T^{2-2/d})``
    integers -- exactly the count that route is meant to expose, and exactly what
    pooling avoids materialising.  Because ``prof`` is round-robin, the columns
    of a profile are an arithmetic progression, so the table is built without a
    sort or a Python loop.
    """
    n_X, N0 = spec.n_X, spec.N0
    g = -(-n_X // N0)                                   # ceil: widest group
    pts = spec.C_indices.reshape(spec.B, -1).astype(np.int64)   # (B, deg)
    base = pts[:, :, None] + np.arange(g, dtype=np.int64)[None, None, :] * N0
    return np.where(base < n_X, base, n_X).reshape(spec.B, -1).astype(np.int32)

"""Backward passes for the three SMAT Triton kernels, as autograd Functions.

The repo's kernels (../smat/smat_triton.py) are forward-only, so training has to
use the torch path.  This module wraps each kernel in a ``torch.autograd.Function``
whose backward is itself fused, and ``patch()`` swaps the wrappers into
``smat_attn``'s kernel table so that ``smat_attention(..., backend="auto")``
becomes differentiable with no change to the repo.

    chunk_local   Out = Phi Sin + tril(Phi Psi^T) Vb          (per chunk)
                  dSin = Phi^T dOut                            (one GEMM, torch)
                  dA   = tril(dOut Vb^T)                       (C x C, in registers)
                  dPhi = dOut Sin^T + dA Psi                   (row-parallel kernel)
                  dPsi = dA^T Phi,   dVb = A^T dOut            (column-parallel kernel)

    pool          F_x = sum_{prof(j)=x} psi_j (x) vb_j
                  dpsi_j = dF_x vb_j,   dvb_j = dF_x^T psi_j   (one kernel, dF_x loaded once)

    incidence     U = C F  ->  dF = C^T dU                     (the forward kernel with
                                                                the transposed table)

Like the forward kernels these compute in fp32 with TF32 dot products and return
fp32, cast back to the input dtype.  They are for training; the fp64 reference
path stays on torch exactly as ``smat_attn._tri_ok`` requires.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smat"))
import smat_triton as _tri          # noqa: E402
import smat_attn                    # noqa: E402

import triton                       # noqa: E402
import triton.language as tl        # noqa: E402

# bind the forward kernels now, before patch() replaces the module attributes
_fwd_chunk_local = _tri.chunk_local
_fwd_pool = _tri.pool
_fwd_incidence = _tri.incidence


# ---------------------------------------------------------------------------
# chunk_local backward
# ---------------------------------------------------------------------------

@triton.jit
def _cl_bwd_dphi_kernel(
    Phi, Psi, Vb, Sin, dOut, dPhi,
    stride_qb, stride_qt, stride_qr,
    stride_vb, stride_vt, stride_vp,
    stride_sb, stride_sr, stride_sp,
    P, R,
    C: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr,
):
    """dPhi_m = dOut_m Sin^T + sum_{n <= m} tril(dOut_m Vb_n^T) Psi_n."""
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_p = tl.arange(0, BLOCK_P)
    offs_r = tl.arange(0, BLOCK_R)
    m_mask = offs_m < C
    p_mask = offs_p < P
    r_mask = offs_r < R

    do = tl.load(dOut + pid_b * stride_vb + offs_m[:, None] * stride_vt
                 + offs_p[None, :] * stride_vp,
                 mask=m_mask[:, None] & p_mask[None, :], other=0.0).to(tl.float32)
    # inter-chunk term: dOut_m Sin^T   (BM,P) x (P,R)
    st = tl.load(Sin + pid_b * stride_sb + offs_r[:, None] * stride_sr
                 + offs_p[None, :] * stride_sp,
                 mask=r_mask[:, None] & p_mask[None, :], other=0.0)      # (R,P)
    acc = tl.dot(do, tl.trans(st), allow_tf32=True)                       # (BM,R)

    n_end = tl.minimum((pid_m + 1) * BLOCK_M, C)
    for n0 in range(0, n_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < C
        v = tl.load(Vb + pid_b * stride_vb + offs_n[:, None] * stride_vt
                    + offs_p[None, :] * stride_vp,
                    mask=n_mask[:, None] & p_mask[None, :], other=0.0).to(tl.float32)
        dA = tl.dot(do, tl.trans(v), allow_tf32=True)                     # (BM,BN)
        dA = tl.where(offs_m[:, None] >= offs_n[None, :], dA, 0.0)
        psi = tl.load(Psi + pid_b * stride_qb + offs_n[:, None] * stride_qt
                      + offs_r[None, :] * stride_qr,
                      mask=n_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(dA, psi, allow_tf32=True)                           # (BM,R)

    tl.store(dPhi + pid_b * stride_qb + offs_m[:, None] * stride_qt
             + offs_r[None, :] * stride_qr, acc,
             mask=m_mask[:, None] & r_mask[None, :])


@triton.jit
def _cl_bwd_dpsi_dvb_kernel(
    Phi, Psi, Vb, dOut, dPsi, dVb,
    stride_qb, stride_qt, stride_qr,
    stride_vb, stride_vt, stride_vp,
    P, R,
    C: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr,
):
    """dPsi_n = sum_{m >= n} dA_{mn}^T Phi_m,   dVb_n = sum_{m >= n} A_{mn}^T dOut_m."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_p = tl.arange(0, BLOCK_P)
    offs_r = tl.arange(0, BLOCK_R)
    n_mask = offs_n < C
    p_mask = offs_p < P
    r_mask = offs_r < R

    psi = tl.load(Psi + pid_b * stride_qb + offs_n[:, None] * stride_qt
                  + offs_r[None, :] * stride_qr,
                  mask=n_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)
    v = tl.load(Vb + pid_b * stride_vb + offs_n[:, None] * stride_vt
                + offs_p[None, :] * stride_vp,
                mask=n_mask[:, None] & p_mask[None, :], other=0.0).to(tl.float32)
    acc_psi = tl.zeros((BLOCK_N, BLOCK_R), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_N, BLOCK_P), dtype=tl.float32)

    m_start = pid_n * BLOCK_N
    for m0 in range(0, C, BLOCK_M):
        if m0 + BLOCK_M > m_start:            # blocks entirely above the diagonal contribute nothing
            offs_m = m0 + tl.arange(0, BLOCK_M)
            m_mask = offs_m < C
            phi = tl.load(Phi + pid_b * stride_qb + offs_m[:, None] * stride_qt
                          + offs_r[None, :] * stride_qr,
                          mask=m_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)
            do = tl.load(dOut + pid_b * stride_vb + offs_m[:, None] * stride_vt
                         + offs_p[None, :] * stride_vp,
                         mask=m_mask[:, None] & p_mask[None, :], other=0.0).to(tl.float32)
            keep = offs_m[:, None] >= offs_n[None, :]
            A = tl.where(keep, tl.dot(phi, tl.trans(psi), allow_tf32=True), 0.0)   # (BM,BN)
            dA = tl.where(keep, tl.dot(do, tl.trans(v), allow_tf32=True), 0.0)     # (BM,BN)
            acc_psi += tl.dot(tl.trans(dA), phi, allow_tf32=True)                  # (BN,R)
            acc_v += tl.dot(tl.trans(A), do, allow_tf32=True)                      # (BN,P)

    tl.store(dPsi + pid_b * stride_qb + offs_n[:, None] * stride_qt
             + offs_r[None, :] * stride_qr, acc_psi,
             mask=n_mask[:, None] & r_mask[None, :])
    tl.store(dVb + pid_b * stride_vb + offs_n[:, None] * stride_vt
             + offs_p[None, :] * stride_vp, acc_v,
             mask=n_mask[:, None] & p_mask[None, :])


def _pow2(x, lo=16, hi=128):
    return min(hi, max(lo, triton.next_power_of_2(x)))


class _ChunkLocalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Phi_c, Psi_c, Vb_c, Sin):
        out = _fwd_chunk_local(Phi_c, Psi_c, Vb_c, Sin)
        ctx.save_for_backward(Phi_c, Psi_c, Vb_c, Sin)
        return out

    @staticmethod
    def backward(ctx, dOut):
        Phi_c, Psi_c, Vb_c, Sin = ctx.saved_tensors
        nb, nc, C, R = Phi_c.shape
        P = Vb_c.shape[-1]
        if R > 128 or P > 128:
            raise NotImplementedError("chunk_local backward kernel assumes r, p <= 128")
        q = Phi_c.contiguous().reshape(nb * nc, C, R)
        k = Psi_c.contiguous().reshape(nb * nc, C, R)
        v = Vb_c.contiguous().reshape(nb * nc, C, P)
        s = Sin.contiguous().float().reshape(nb * nc, R, P)
        do = dOut.contiguous().float().reshape(nb * nc, C, P)

        dq = torch.empty(nb * nc, C, R, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dv = torch.empty(nb * nc, C, P, device=q.device, dtype=torch.float32)
        bm = bn = _pow2(C, hi=64)
        bp, br = _pow2(P), _pow2(R)
        _cl_bwd_dphi_kernel[(nb * nc, triton.cdiv(C, bm))](
            q, k, v, s, do, dq,
            q.stride(0), q.stride(1), q.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            s.stride(0), s.stride(1), s.stride(2),
            P, R, C=C, BLOCK_M=bm, BLOCK_N=bn, BLOCK_P=bp, BLOCK_R=br,
            num_warps=4, num_stages=2)
        _cl_bwd_dpsi_dvb_kernel[(nb * nc, triton.cdiv(C, bn))](
            q, k, v, do, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            P, R, C=C, BLOCK_M=bm, BLOCK_N=bn, BLOCK_P=bp, BLOCK_R=br,
            num_warps=4, num_stages=2)
        ds = q.float().transpose(-1, -2) @ do                              # (nb*nc, R, P)
        return (dq.reshape(nb, nc, C, R).to(Phi_c.dtype),
                dk.reshape(nb, nc, C, R).to(Psi_c.dtype),
                dv.reshape(nb, nc, C, P).to(Vb_c.dtype),
                ds.reshape(nb, nc, R, P).to(Sin.dtype))


# ---------------------------------------------------------------------------
# pool backward
# ---------------------------------------------------------------------------

@triton.jit
def _pool_bwd_kernel(
    Psi, Vb, dF, dPsi, dVb,
    stride_qb, stride_qt, stride_qr,
    stride_vb, stride_vt, stride_vp,
    stride_fb, stride_fn, stride_fr,
    NX, N0, N_P, R, P,
    BLOCK_S: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_P: tl.constexpr,
):
    """For every column j of profile x: dpsi_j = dF_x vb_j, dvb_j = dF_x^T psi_j."""
    x = tl.program_id(0)
    b = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_r = tl.arange(0, BLOCK_R)
    offs_p = tl.arange(0, BLOCK_P)
    r_mask = offs_r < R
    p_mask = offs_p < P
    df = tl.load(dF + b * stride_fb + x * stride_fn + offs_r[:, None] * stride_fr
                 + offs_p[None, :], mask=r_mask[:, None] & p_mask[None, :],
                 other=0.0).to(tl.float32)                                 # (R,P)

    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rel = s * N0 + x
    s_mask = rel < NX
    col = N_P + rel
    psi = tl.load(Psi + b * stride_qb + col[:, None] * stride_qt + offs_r[None, :] * stride_qr,
                  mask=s_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)   # (S,R)
    v = tl.load(Vb + b * stride_vb + col[:, None] * stride_vt + offs_p[None, :] * stride_vp,
                mask=s_mask[:, None] & p_mask[None, :], other=0.0).to(tl.float32)     # (S,P)
    dpsi = tl.dot(v, tl.trans(df), allow_tf32=True)                       # (S,R)
    dv = tl.dot(psi, df, allow_tf32=True)                                 # (S,P)
    tl.store(dPsi + b * stride_qb + col[:, None] * stride_qt + offs_r[None, :] * stride_qr,
             dpsi, mask=s_mask[:, None] & r_mask[None, :])
    tl.store(dVb + b * stride_vb + col[:, None] * stride_vt + offs_p[None, :] * stride_vp,
             dv, mask=s_mask[:, None] & p_mask[None, :])


class _PoolFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Psi, Vb, n_P, n, N0):
        F = _fwd_pool(Psi, Vb, n_P, n, N0)
        ctx.save_for_backward(Psi, Vb)
        ctx.dims = (n_P, n, N0)
        return F

    @staticmethod
    def backward(ctx, dF):
        Psi, Vb = ctx.saved_tensors
        n_P, n, N0 = ctx.dims
        nb, T, R = Psi.shape
        P = Vb.shape[-1]
        if R > 128 or P > 128:
            raise NotImplementedError("pool backward kernel assumes r, p <= 128")
        n_X = n - n_P
        gmax = -(-n_X // N0)
        Psi_c, Vb_c = Psi.contiguous(), Vb.contiguous()
        dFc = dF.contiguous().float()
        dPsi = torch.zeros(nb, T, R, device=Psi.device, dtype=torch.float32)
        dVb = torch.zeros(nb, T, P, device=Psi.device, dtype=torch.float32)
        bs = _pow2(gmax, hi=64)
        _pool_bwd_kernel[(N0, nb, triton.cdiv(gmax, bs))](
            Psi_c, Vb_c, dFc, dPsi, dVb,
            Psi_c.stride(0), Psi_c.stride(1), Psi_c.stride(2),
            Vb_c.stride(0), Vb_c.stride(1), Vb_c.stride(2),
            dFc.stride(0), dFc.stride(1), dFc.stride(2),
            n_X, N0, n_P, R, P,
            BLOCK_S=bs, BLOCK_R=_pow2(R), BLOCK_P=_pow2(P),
            num_warps=4, num_stages=2)
        return dPsi.to(Psi.dtype), dVb.to(Vb.dtype), None, None, None


# ---------------------------------------------------------------------------
# incidence backward: the same kernel with the transposed table
# ---------------------------------------------------------------------------

_TRANSPOSE_CACHE = {}


def _point_planes(plane_pts, N0):
    """(N0, degT) table of the planes through each point, from the (B, deg) table."""
    key = (plane_pts.data_ptr(), tuple(plane_pts.shape), N0)
    hit = _TRANSPOSE_CACHE.get(key)
    if hit is not None:
        return hit
    B, deg = plane_pts.shape
    pts = plane_pts.reshape(-1).long()
    planes = torch.arange(B, device=plane_pts.device).repeat_interleave(deg)
    order = torch.argsort(pts, stable=True)
    pts, planes = pts[order], planes[order]
    counts = torch.bincount(pts, minlength=N0)
    if counts.min() != counts.max():
        raise ValueError("incidence transpose is not uniform; cannot reuse the kernel")
    table = planes.reshape(N0, int(counts[0])).to(plane_pts.dtype).contiguous()
    _TRANSPOSE_CACHE[key] = table
    return table


class _IncidenceFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, F, plane_pts, B):
        U = _fwd_incidence(F, plane_pts, B)
        ctx.table = _point_planes(plane_pts, F.shape[1])
        ctx.N0 = F.shape[1]
        ctx.in_dtype = F.dtype
        return U

    @staticmethod
    def backward(ctx, dU):
        dF = _fwd_incidence(dU.contiguous(), ctx.table, ctx.N0)
        return dF.to(ctx.in_dtype), None, None


# ---------------------------------------------------------------------------
# the differentiable entry points, and the patch
# ---------------------------------------------------------------------------

def chunk_local(Phi_c, Psi_c, Vb_c, Sin):
    return _ChunkLocalFn.apply(Phi_c, Psi_c, Vb_c, Sin)


def pool(Psi, Vb, n_P, n, N0):
    return _PoolFn.apply(Psi, Vb, n_P, n, N0)


def incidence(F, plane_pts, B):
    return _IncidenceFn.apply(F, plane_pts, B)


_ORIG = {}


def patch():
    """Swap the differentiable wrappers into smat_attn's kernel table."""
    if _ORIG:
        return
    for name, fn in (("chunk_local", chunk_local), ("pool", pool), ("incidence", incidence)):
        _ORIG[name] = getattr(_tri, name)
        setattr(_tri, name, fn)


def unpatch():
    for name, fn in _ORIG.items():
        setattr(_tri, name, fn)
    _ORIG.clear()

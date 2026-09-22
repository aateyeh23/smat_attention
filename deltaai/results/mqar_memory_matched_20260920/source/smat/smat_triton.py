#!/usr/bin/env python3
"""Triton kernels for the two passes that dominate SMAT prefill on a GPU.

Both exist because the obvious torch expression materialises an intermediate
that is asymptotically larger than the work itself.

``chunk_local``
    The within-chunk term of Eq. (3.19), fused with the inter-chunk state
    product.  The torch form builds the score tile ``(nb, nc, C, C)``
    explicitly -- 1.1 GB at ``T = 2^18, nb = 8, C = 128`` -- whereas Sec. 3.3
    promises the tile "never leaves on-chip memory".  The kernel keeps it in
    registers and writes only the ``(nb, T, p)`` output.

``incidence``
    ``U_h = sum_{x in H_h} F_x``, the application of ``C``.  The torch form
    ``F[:, plane_pts].sum(2)`` allocates ``nnz(C) * r * p`` words -- 35 GB at
    ``d = 3, T = 2^18`` -- to perform ``nnz(C) * r * p`` adds.  The kernel holds
    the accumulator in registers, so the traffic is the ``nnz(C) * r * p`` reads
    the cost model actually charges for, and nothing else.

Both fall back to the torch path in ``smat_attn`` when Triton is unavailable.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAVE = True
except Exception:                                       # pragma: no cover
    triton = None
    tl = None
    _HAVE = False


def available() -> bool:
    return _HAVE and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# fused chunk-local pass
# ---------------------------------------------------------------------------

if _HAVE:

    @triton.jit
    def _chunk_local_kernel(
        Phi, Psi, Vb, Sin, Out,
        stride_qb, stride_qt, stride_qr,
        stride_vb, stride_vt, stride_vp,
        stride_sb, stride_sr, stride_sp,
        P,
        C: tl.constexpr, R: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr,
    ):
        in_dt = Phi.dtype.element_ty      # bf16 / fp16 / fp32 as given
        pid_b = tl.program_id(0)          # flattened (batch, chunk)
        pid_m = tl.program_id(1)
        pid_p = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        m_mask = offs_m < C
        p_mask = offs_p < P

        acc = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)

        # ---- inter-chunk: Phi_b S^in_b, in fp32 so the carried state keeps
        # its precision (it is a sum over every earlier chunk of the segment).
        for r0 in range(0, R, BLOCK_R):
            offs_r = r0 + tl.arange(0, BLOCK_R)
            r_mask = offs_r < R
            phi = tl.load(
                Phi + pid_b * stride_qb + offs_m[:, None] * stride_qt
                + offs_r[None, :] * stride_qr,
                mask=m_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)
            st = tl.load(
                Sin + pid_b * stride_sb + offs_r[:, None] * stride_sr
                + offs_p[None, :] * stride_sp,
                mask=r_mask[:, None] & p_mask[None, :], other=0.0)
            acc += tl.dot(phi, st, allow_tf32=True)

        # ---- within-chunk: tril(Phi_b Psi_b^T) Vb_b, tile never leaves SRAM
        n_end = tl.minimum((pid_m + 1) * BLOCK_M, C)
        for n0 in range(0, n_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < C
            s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for r0 in range(0, R, BLOCK_R):
                offs_r = r0 + tl.arange(0, BLOCK_R)
                r_mask = offs_r < R
                phi = tl.load(
                    Phi + pid_b * stride_qb + offs_m[:, None] * stride_qt
                    + offs_r[None, :] * stride_qr,
                    mask=m_mask[:, None] & r_mask[None, :], other=0.0)
                psi = tl.load(
                    Psi + pid_b * stride_qb + offs_n[:, None] * stride_qt
                    + offs_r[None, :] * stride_qr,
                    mask=n_mask[:, None] & r_mask[None, :], other=0.0)
                s += tl.dot(phi.to(in_dt), tl.trans(psi.to(in_dt)),
                            allow_tf32=True)
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, 0.0)
            v = tl.load(
                Vb + pid_b * stride_vb + offs_n[:, None] * stride_vt
                + offs_p[None, :] * stride_vp,
                mask=n_mask[:, None] & p_mask[None, :], other=0.0)
            acc += tl.dot(s.to(in_dt), v.to(in_dt), allow_tf32=True)

        tl.store(
            Out + pid_b * stride_vb + offs_m[:, None] * stride_vt
            + offs_p[None, :] * stride_vp,
            acc, mask=m_mask[:, None] & p_mask[None, :])


def chunk_local(Phi_c, Psi_c, Vb_c, Sin, *, block_m: int = 64,
                block_n: int = 64, block_p: int = 64, block_r: int = 64):
    """``Phi_b S^in_b + (Phi_b Psi_b^T . L_C) Vb_b`` for every chunk, fused.

    Shapes ``(nb, nc, C, r)``, ``(nb, nc, C, r)``, ``(nb, nc, C, p)``,
    ``(nb, nc, r, p)`` -> ``(nb, nc, C, p)``.
    """
    nb, nc, C, R = Phi_c.shape
    P = Vb_c.shape[-1]
    Phi_c = Phi_c.contiguous()
    Psi_c = Psi_c.contiguous()
    Vb_c = Vb_c.contiguous()
    Sin = Sin.contiguous().float()

    q = Phi_c.reshape(nb * nc, C, R)
    k = Psi_c.reshape(nb * nc, C, R)
    v = Vb_c.reshape(nb * nc, C, P)
    s = Sin.reshape(nb * nc, R, P)
    out = torch.empty(nb * nc, C, P, device=q.device, dtype=torch.float32)

    bm = min(block_m, max(16, triton.next_power_of_2(C)))
    bn = min(block_n, max(16, triton.next_power_of_2(C)))
    bp = min(block_p, max(16, triton.next_power_of_2(P)))
    br = min(block_r, max(16, triton.next_power_of_2(R)))
    grid = (nb * nc, triton.cdiv(C, bm), triton.cdiv(P, bp))
    _chunk_local_kernel[grid](
        q, k, v, s, out,
        q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        P, C=C, R=R,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_P=bp, BLOCK_R=br,
        num_warps=4, num_stages=2,
    )
    return out.reshape(nb, nc, C, P)


# ---------------------------------------------------------------------------
# incidence contraction U = C F
# ---------------------------------------------------------------------------

if _HAVE:

    @triton.jit
    def _incidence_kernel(
        F, PlanePts, U,
        stride_fb, stride_fn,
        stride_ub, stride_uh,
        RP,
        DEG: tl.constexpr, BLOCK: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        h = tl.program_id(0)
        b = tl.program_id(1)
        pid_c = tl.program_id(2)

        offs = pid_c * BLOCK + tl.arange(0, BLOCK)
        mask = offs < RP
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

        # DEG is the uniform row degree q^{d-2}, passed as a constexpr: Triton's
        # frontend cannot compile a range() with runtime bounds under Python
        # 3.12+, and a constant trip count is faster anyway.
        for t0 in range(0, DEG, BLOCK_D):
            d_offs = t0 + tl.arange(0, BLOCK_D)
            dm = d_offs < DEG
            xs = tl.load(PlanePts + h * DEG + d_offs, mask=dm, other=0).to(tl.int64)
            vals = tl.load(F + b * stride_fb + xs[:, None] * stride_fn
                           + offs[None, :],
                           mask=dm[:, None] & mask[None, :], other=0.0)
            acc += tl.sum(vals, axis=0)
        tl.store(U + b * stride_ub + h * stride_uh + offs, acc, mask=mask)


def incidence(F, plane_pts, B: int, *, block: int = 128, block_d: int = 16):
    """``U_h = sum_{x in H_h} F_x`` with the accumulator held in registers.

    ``F: (nb, N0, r, p) -> U: (nb, B, r, p)``, ``plane_pts: (B, deg)``.

    The ``rp`` tile is the slowest grid axis so that concurrently resident CTAs
    all read the same slice of ``F``.  That slice is ``nb * N0 * BLOCK`` floats
    -- a few MB -- so each ``F`` element crosses HBM once and is served from L2
    for its remaining ``deg * B / N0`` uses.  The torch form instead streams the
    whole of ``F`` once per degree step and materialises the gather.
    """
    nb, N0, R, P = F.shape
    RP = R * P
    deg = plane_pts.shape[1]
    Ff = F.contiguous().float().reshape(nb, N0, RP)
    U = torch.empty(nb, B, RP, device=F.device, dtype=torch.float32)
    bd = min(block_d, max(1, triton.next_power_of_2(deg)))
    grid = (B, nb, triton.cdiv(RP, block))
    _incidence_kernel[grid](
        Ff, plane_pts, U,
        Ff.stride(0), Ff.stride(1),
        U.stride(0), U.stride(1),
        RP, DEG=deg, BLOCK=block, BLOCK_D=bd,
        num_warps=4, num_stages=2,
    )
    return U.reshape(nb, B, R, P)


# ---------------------------------------------------------------------------
# profile pooling  F = S^T Z
# ---------------------------------------------------------------------------

if _HAVE:

    @triton.jit
    def _pool_kernel(
        Psi, Vb, F,
        stride_qb, stride_qt, stride_qr,
        stride_vb, stride_vt, stride_vp,
        stride_fb, stride_fn, stride_fr,
        NX, N0, N_P, R, P,
        GMAX: tl.constexpr, BLOCK_S: tl.constexpr,
        BLOCK_R: tl.constexpr, BLOCK_P: tl.constexpr,
    ):
        in_dt = Psi.dtype.element_ty
        x = tl.program_id(0)              # profile index
        b = tl.program_id(1)              # batch * head
        pid_r = tl.program_id(2)

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_p = tl.arange(0, BLOCK_P)
        r_mask = offs_r < R
        p_mask = offs_p < P
        acc = tl.zeros((BLOCK_R, BLOCK_P), dtype=tl.float32)

        # prof(j) = (j - N_P) mod N0, so this profile's columns are the
        # arithmetic progression x, x + N0, x + 2 N0, ... -- addressed directly,
        # with no gather and no permuted copy of the whole activation.
        # NB the three-argument form: Triton's frontend cannot compile a
        # two-argument range() under Python 3.12+ (it reaches for ast.Num,
        # removed in 3.12), and cannot compile runtime bounds at all.
        for s0 in range(0, GMAX, BLOCK_S):
            s = s0 + tl.arange(0, BLOCK_S)
            rel = s * N0 + x
            s_mask = rel < NX
            col = N_P + rel
            psi = tl.load(Psi + b * stride_qb + col[:, None] * stride_qt
                          + offs_r[None, :] * stride_qr,
                          mask=s_mask[:, None] & r_mask[None, :], other=0.0)
            v = tl.load(Vb + b * stride_vb + col[:, None] * stride_vt
                        + offs_p[None, :] * stride_vp,
                        mask=s_mask[:, None] & p_mask[None, :], other=0.0)
            acc += tl.dot(tl.trans(psi).to(in_dt), v.to(in_dt), allow_tf32=True)

        tl.store(F + b * stride_fb + x * stride_fn + offs_r[:, None] * stride_fr
                 + offs_p[None, :], acc,
                 mask=r_mask[:, None] & p_mask[None, :])


def pool_ok(N0: int, gmax: int, P: int) -> bool:
    """Whether the fused pooling kernel is the right tool for this shape.

    It wins when there are many profiles (so the grid fills the GPU) and each
    group is short.  At small ``N0`` -- ``d = 1`` above all, where ``N0 = 1`` --
    the torch route is a single large GEMM and is far better, so we say so
    rather than forcing the kernel everywhere.
    """
    return _HAVE and N0 >= 64 and gmax <= 8192 and P <= 128


def pool(Psi, Vb, n_P: int, n: int, N0: int, *, block_s: int = 16,
         block_r: int = 64):
    """``F_x = sum_{prof(j)=x} Psi_j Vb_j^T`` for every profile.

    ``(nb, T, r), (nb, T, p) -> (nb, N0, r, p)``.  The torch form decomposes
    into ``N0`` batched GEMMs of shape ``(r x g)(g x p)`` with ``g = n_X/N0``
    around 20, which is small enough that cuBLAS spends its time on launch
    rather than on arithmetic; this walks the same progressions with the
    accumulator in registers.
    """
    nb, T, R = Psi.shape
    P = Vb.shape[-1]
    n_X = n - n_P
    gmax = -(-n_X // N0)
    Psi = Psi.contiguous()
    Vb = Vb.contiguous()
    F = torch.empty(nb, N0, R, P, device=Psi.device, dtype=torch.float32)
    bs = max(16, triton.next_power_of_2(min(block_s, max(1, gmax))))
    bp = max(16, triton.next_power_of_2(P))
    br = min(block_r, max(16, triton.next_power_of_2(R)))
    grid = (N0, nb, triton.cdiv(R, br))
    _pool_kernel[grid](
        Psi, Vb, F,
        Psi.stride(0), Psi.stride(1), Psi.stride(2),
        Vb.stride(0), Vb.stride(1), Vb.stride(2),
        F.stride(0), F.stride(1), F.stride(2),
        n_X, N0, n_P, R, P,
        GMAX=triton.cdiv(gmax, bs) * bs, BLOCK_S=bs, BLOCK_R=br, BLOCK_P=bp,
        num_warps=8, num_stages=2,
    )
    return F

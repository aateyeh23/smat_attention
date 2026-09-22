"""Benchmarked aggregation/projection alternatives; production retains cuBLAS."""
import torch
import triton
import triton.language as tl


class _Incidence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, states, plane_points, point_planes):
        from smat_triton import incidence as kernel
        ctx.save_for_backward(point_planes)
        ctx.n = states.shape[1]
        return kernel(states, plane_points, plane_points.shape[0])

    @staticmethod
    def backward(ctx, grad):
        from smat_triton import incidence as kernel
        point_planes, = ctx.saved_tensors
        return kernel(grad.contiguous(), point_planes, ctx.n), None, None


def incidence(states, plane_points, point_planes):
    return _Incidence.apply(states, plane_points, point_planes)


@triton.jit
def _incidence_mm(A, F, Y, NI: tl.constexpr, NO: tl.constexpr, RP: tl.constexpr,
                  AS0: tl.constexpr, AS1: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    mi, ni, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mi*BM+tl.arange(0, BM)
    n = ni*BN+tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(NI, BK)):
        kk = start*BK+k
        a = tl.load(A+m[:, None]*AS0+kk[None, :]*AS1,
                    (m<NO)[:, None] & (kk<NI)[None, :], other=0)
        f = tl.load(F+batch*NI*RP+kk[:, None]*RP+n[None, :],
                    (kk<NI)[:, None] & (n<RP)[None, :], other=0)
        acc = tl.dot(a, f, acc, input_precision='tf32x3')
    tl.store(Y+batch*NO*RP+m[:, None]*RP+n[None, :], acc,
             (m<NO)[:, None] & (n<RP)[None, :])


def incidence_mm(states, matrix):
    b, ni, r, p = states.shape
    no = matrix.shape[0]
    output = states.new_empty(b, no, r, p)
    _incidence_mm[(triton.cdiv(no, 32), triton.cdiv(r*p, 64), b)](
        matrix, states.contiguous(), output, ni, no, r*p, *matrix.stride(),
        32, 64, 32, num_warps=4, num_stages=3)
    return output


class _DenseIncidence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, states, matrix):
        ctx.save_for_backward(matrix)
        return incidence_mm(states, matrix)

    @staticmethod
    def backward(ctx, grad):
        matrix, = ctx.saved_tensors
        return incidence_mm(grad, matrix.T), None


def dense_incidence(states, matrix):
    return _DenseIncidence.apply(states, matrix)


@triton.jit
def _mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        A0: tl.constexpr, A1: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Group nearby output rows to reuse B tiles in L2.
    pid = tl.program_id(0)
    n_m, n_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid//(8*n_n)
    first_m = group*8
    group_m = tl.minimum(n_m-first_m, 8)
    mi = first_m+(pid % (8*n_n)) % group_m
    ni = (pid % (8*n_n))//group_m
    m, n, k = mi*BM+tl.arange(0, BM), ni*BN+tl.arange(0, BN), tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start*BK+k
        a = tl.load(A+m[:, None]*A0+kk[None, :]*A1,
                    (m<M)[:, None] & (kk<K)[None, :], other=0)
        b = tl.load(B+kk[:, None]*B0+n[None, :]*B1,
                    (kk<K)[:, None] & (n<N)[None, :], other=0)
        acc = tl.dot(a, b, acc, input_precision='tf32x3')
    tl.store(C+m[:, None]*N+n[None, :], acc, (m<M)[:, None] & (n<N)[None, :])


def mm(a, b):
    m, k = a.shape
    n = b.shape[1]
    assert b.shape[0] == k
    c = a.new_empty(m, n)
    _mm[(triton.cdiv(m, 64)*triton.cdiv(n, 64),)](
        a, b, c, m, n, k, *a.stride(), *b.stride(), 64, 64, 32,
        num_warps=4, num_stages=3)
    return c


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        b, length, width = x.shape
        h, n, _ = weight.shape
        xf, wf = x.reshape(-1, width).contiguous(), weight.reshape(-1, width).contiguous()
        ctx.save_for_backward(xf, wf)
        ctx.shape = (b, length, width, h, n)
        return mm(xf, wf.T).reshape(b, length, h, n).permute(0, 2, 1, 3)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        b, length, width, h, n = ctx.shape
        dy = grad.permute(0, 2, 1, 3).reshape(b*length, h*n).contiguous()
        dx = mm(dy, w).reshape(b, length, width)
        dw = mm(dy.T, x).reshape(h, n, width)
        return dx, dw


def read_projection(x, weight):
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError('Fused projection requires float32')
    return _Projection.apply(x, weight)

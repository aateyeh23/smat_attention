"""Grouped top-k memory reads with fused Triton forward and backward.

Routes are sorted once on GPU. Kernels gather directly into tensor-core tiles;
there are no padded query/state tensors or data-dependent host synchronizations.
State gradients reduce each cell's sorted readers without floating-point atomics.
The float32 path uses tf32x3 dot products, matching the training precision recipe.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _top4_fwd(X, Bias, Idx, W, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
              S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
              BN: tl.constexpr):
    row = tl.program_id(0)
    b, h, m = row//(H*M), (row//M)%H, row%M
    n = tl.arange(0, BN)
    x = tl.load(X+b*S0+h*S1+m*S2+n*S3, n<N, other=-float('inf'))
    x += tl.load(Bias+h*N+n, n<N, other=0)
    best = tl.max(x, 0)
    i0 = tl.min(tl.where(x == best, n, 2147483647), 0)
    x = tl.where(n == i0, -float('inf'), x)
    s1 = tl.max(x, 0)
    i1 = tl.min(tl.where(x == s1, n, 2147483647), 0)
    x = tl.where(n == i1, -float('inf'), x)
    s2 = tl.max(x, 0)
    i2 = tl.min(tl.where(x == s2, n, 2147483647), 0)
    x = tl.where(n == i2, -float('inf'), x)
    s3 = tl.max(x, 0)
    i3 = tl.min(tl.where(x == s3, n, 2147483647), 0)
    e1, e2, e3 = tl.exp(s1-best), tl.exp(s2-best), tl.exp(s3-best)
    z = 1+e1+e2+e3
    tl.store(Idx+row*4, i0)
    tl.store(Idx+row*4+1, i1)
    tl.store(Idx+row*4+2, i2)
    tl.store(Idx+row*4+3, i3)
    tl.store(W+row*4, 1/z)
    tl.store(W+row*4+1, e1/z)
    tl.store(W+row*4+2, e2/z)
    tl.store(W+row*4+3, e3/z)


@triton.jit
def _top4_bwd(Idx, W, DW, DX, N: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, 4)
    n = tl.arange(0, BN)
    idx = tl.load(Idx+row*4+k)
    w = tl.load(W+row*4+k)
    dw = tl.load(DW+row*4+k)
    ds = w*(dw-tl.sum(w*dw, 0))
    dx = tl.sum(tl.where(n[:, None] == idx[None, :], ds[None, :], 0), 1)
    tl.store(DX+row*N+n, dx, n<N)


class _Top4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, bias):
        b, h, m, n = logits.shape
        idx = torch.empty(b*h, m, 4, device=logits.device, dtype=torch.int64)
        w = torch.empty(b*h, m, 4, device=logits.device, dtype=torch.float32)
        _top4_fwd[(b*h*m,)](logits, bias, idx, w, h, m, n, *logits.stride(),
                            triton.next_power_of_2(n), num_warps=4)
        ctx.save_for_backward(idx, w)
        ctx.shape = logits.shape
        ctx.mark_non_differentiable(idx)
        return idx, w

    @staticmethod
    def backward(ctx, _, dw):
        idx, w = ctx.saved_tensors
        b, h, m, n = ctx.shape
        dx = w.new_empty(b, h, m, n)
        _top4_bwd[(b*h*m,)](idx, w, dw.contiguous(), dx, n,
                            triton.next_power_of_2(n), num_warps=4)
        return dx, dx.sum((0, 2))


def top4_softmax(logits, bias):
    """Fuse bias, distinct top-4 and softmax; ties prefer the lowest type index."""
    if logits.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError('Fused router requires float32')
    if logits.shape[-1] < 4:
        raise ValueError('Need at least four read types')
    return _Top4.apply(logits, bias.contiguous())


@triton.jit
def _route_keys(Idx, Keys, Counts, TOTAL: tl.constexpr, QK: tl.constexpr,
                N: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    cell = (e//QK)*N + tl.load(Idx+e, e<TOTAL, other=0).to(tl.int32)
    tl.store(Keys+e, cell, e<TOTAL)
    tl.atomic_add(Counts+cell, 1, e<TOTAL, sem='relaxed')


@triton.jit
def _find_cell(Substarts, group, NC: tl.constexpr):
    lo = 0
    hi = NC
    while lo < hi:
        mid = (lo+hi)//2
        stop = tl.load(Substarts+mid+1)
        go_right = stop <= group
        lo = tl.where(go_right, mid+1, lo)
        hi = tl.where(go_right, hi, mid)
    return lo


@triton.jit
def _group_fwd(Q, W, F, Order, Starts, Substarts, Ypairs,
               NC: tl.constexpr, K: tl.constexpr, R: tl.constexpr, P: tl.constexpr,
               BM: tl.constexpr, BR: tl.constexpr, BP: tl.constexpr):
    group = tl.program_id(0)
    if group < tl.load(Substarts+NC):
        cell = _find_cell(Substarts, group, NC)
        start = tl.load(Starts+cell)
        stop = tl.load(Starts+cell+1)
        row = start + (group-tl.load(Substarts+cell))*BM + tl.arange(0, BM)
        ev = tl.load(Order+row, row<stop, other=0)
        r = tl.arange(0, BR)
        p = tl.arange(0, BP)
        q = tl.load(Q+(ev//K)[:, None]*R+r[None, :],
                    (row<stop)[:, None] & (r<R)[None, :], other=0).to(tl.float32)
        w = tl.load(W+ev, row<stop, other=0).to(tl.float32)
        f = tl.load(F+cell*R*P+r[:, None]*P+p[None, :],
                    (r<R)[:, None] & (p<P)[None, :], other=0).to(tl.float32)
        y = tl.dot(q*w[:, None], f, input_precision='tf32x3')
        tl.store(Ypairs+ev[:, None]*P+p[None, :], y,
                 (row<stop)[:, None] & (p<P)[None, :])


@triton.jit
def _reduce_pairs(Pairs, Out, K: tl.constexpr, C: tl.constexpr,
                  BK: tl.constexpr, BC: tl.constexpr):
    q = tl.program_id(0)
    k = tl.arange(0, BK)
    c = tl.arange(0, BC)
    x = tl.load(Pairs+(q*K+k[:, None])*C+c[None, :],
                (k<K)[:, None] & (c<C)[None, :], other=0)
    tl.store(Out+q*C+c, tl.sum(x, axis=0), c<C)


@triton.jit
def _group_bwd_q(Q, W, F, DY, Order, Starts, Substarts, DQpairs, DW,
                 NC: tl.constexpr, K: tl.constexpr, R: tl.constexpr, P: tl.constexpr,
                 BM: tl.constexpr, BR: tl.constexpr, BP: tl.constexpr):
    group = tl.program_id(0)
    if group < tl.load(Substarts+NC):
        cell = _find_cell(Substarts, group, NC)
        start = tl.load(Starts+cell)
        stop = tl.load(Starts+cell+1)
        row = start + (group-tl.load(Substarts+cell))*BM + tl.arange(0, BM)
        ev = tl.load(Order+row, row<stop, other=0)
        r = tl.arange(0, BR)
        p = tl.arange(0, BP)
        q = tl.load(Q+(ev//K)[:, None]*R+r[None, :],
                    (row<stop)[:, None] & (r<R)[None, :], other=0).to(tl.float32)
        dy = tl.load(DY+(ev//K)[:, None]*P+p[None, :],
                     (row<stop)[:, None] & (p<P)[None, :], other=0).to(tl.float32)
        f = tl.load(F+cell*R*P+r[:, None]*P+p[None, :],
                    (r<R)[:, None] & (p<P)[None, :], other=0).to(tl.float32)
        w = tl.load(W+ev, row<stop, other=0).to(tl.float32)
        gq = tl.dot(dy, tl.trans(f), input_precision='tf32x3')
        tl.store(DQpairs+ev[:, None]*R+r[None, :], gq*w[:, None],
                 (row<stop)[:, None] & (r<R)[None, :])
        tl.store(DW+ev, tl.sum(gq*q, axis=1), row<stop)


@triton.jit
def _group_bwd_state(Q, W, DY, Order, Starts, DF,
                     K: tl.constexpr, R: tl.constexpr, P: tl.constexpr,
                     BM: tl.constexpr, BR: tl.constexpr, BP: tl.constexpr):
    cell = tl.program_id(0)
    start = tl.load(Starts+cell)
    stop = tl.load(Starts+cell+1)
    r = tl.arange(0, BR)
    p = tl.arange(0, BP)
    acc = tl.zeros((BR, BP), tl.float32)
    for offset in range(start, stop, BM):
        row = offset+tl.arange(0, BM)
        ev = tl.load(Order+row, row<stop, other=0)
        q = tl.load(Q+(ev//K)[:, None]*R+r[None, :],
                    (row<stop)[:, None] & (r<R)[None, :], other=0).to(tl.float32)
        dy = tl.load(DY+(ev//K)[:, None]*P+p[None, :],
                     (row<stop)[:, None] & (p<P)[None, :], other=0).to(tl.float32)
        w = tl.load(W+ev, row<stop, other=0).to(tl.float32)
        acc += tl.dot(tl.trans(q*w[:, None]), dy, input_precision='tf32x3')
    tl.store(DF+cell*R*P+r[:, None]*P+p[None, :], acc,
             (r<R)[:, None] & (p<P)[None, :])


def _layout(idx, n, bm):
    b, q, k = idx.shape
    nc = b*n
    total = idx.numel()
    keys = torch.empty(total, dtype=torch.int32, device=idx.device)
    counts = torch.zeros(nc, dtype=torch.int32, device=idx.device)
    _route_keys[(triton.cdiv(total, 256),)](idx, keys, counts, total, q*k, n, 256)
    order = torch.argsort(keys, stable=True)
    zero = counts.new_zeros(1)
    starts = torch.cat((zero, counts.cumsum(0, dtype=torch.int32)))
    substarts = torch.cat((zero, ((counts+bm-1)//bm).cumsum(0, dtype=torch.int32)))
    # sum(ceil(count/bm)) <= floor(total/bm) + number_of_cells.
    return order, starts, substarts, total//bm+nc


class _Read(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, idx, w, f):
        b, m, r = q.shape
        n, p, k = f.shape[1], f.shape[-1], idx.shape[-1]
        bm, br, bp = 32, max(16, triton.next_power_of_2(r)), max(16, triton.next_power_of_2(p))
        order, starts, substarts, groups = _layout(idx, n, bm)
        pairs = q.new_empty(b, m, k, p)
        y = q.new_empty(b, m, p)
        _group_fwd[(groups,)](q, w, f, order, starts, substarts, pairs,
                              b*n, k, r, p, bm, br, bp, num_warps=4)
        _reduce_pairs[(b*m,)](pairs, y, k, p, triton.next_power_of_2(k),
                              triton.next_power_of_2(p), num_warps=4)
        ctx.save_for_backward(q, w, f, order, starts, substarts)
        ctx.shape = (b, m, n, k, r, p, bm, br, bp, groups)
        return y

    @staticmethod
    def backward(ctx, dy):
        q, w, f, order, starts, substarts = ctx.saved_tensors
        b, m, n, k, r, p, bm, br, bp, groups = ctx.shape
        dy = dy.contiguous()
        pairs = q.new_empty(b, m, k, r)
        dq, dw, df = torch.empty_like(q), torch.empty_like(w), torch.empty_like(f)
        _group_bwd_q[(groups,)](q, w, f, dy, order, starts, substarts, pairs, dw,
                                b*n, k, r, p, bm, br, bp, num_warps=4)
        _reduce_pairs[(b*m,)](pairs, dq, k, r, triton.next_power_of_2(k),
                              triton.next_power_of_2(r), num_warps=4)
        _group_bwd_state[(b*n,)](q, w, dy, order, starts, df,
                                 k, r, p, bm, br, bp, num_warps=4)
        return dq, None, dw, df


def read_topk(q, idx, weights, memory):
    if not q.is_cuda or any(x.dtype != torch.float32 for x in (q, weights, memory)):
        raise ValueError('Fused reader currently supports CUDA float32 only')
    if q.shape[-1] > 128 or memory.shape[-1] > 128:
        raise ValueError('Fused reader supports state dimensions up to 128')
    return _Read.apply(q.contiguous(), idx.contiguous(), weights.contiguous(), memory.contiguous())


"""Memory-bounded segmented ops for the content-addressed SMAT branch (point read).

pool_segmented(Psi, Vb, idx, w, N0)    F[b, x] = sum_{j, k: idx[b,j,k]=x} w[b,j,k] * Psi[b,j] Vb[b,j]^T     -> (B, N0, r, p)
read_gather(Phi, idx, w, F)            y[b, i] = sum_k w[b,i,k] * Phi[b,i]^T F[b, idx[b,i,k]]              -> (B, m, p)

idx/w are the K = 2^dim (cell, weight) pairs per token produced by the straight-through hash (hard forward: one
nonzero; soft phase: interpolation weights).  Both ops are custom autograd Functions: the forward never
materialises per-token outer products for the whole sequence (chunks of --chunk tokens), and the backward
recomputes per chunk from F / F-bar, so peak memory is O(chunk * r * p) instead of O(n * r * p) and there is no
O(N0) factor anywhere.  Exact w.r.t. the dense reference (content_addr.ContentAssign) -- see test below.
"""
import torch

CHUNK = 1024


class _Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Psi, Vb, idx, w, N0):
        B, n, r = Psi.shape; p = Vb.shape[-1]; K = idx.shape[-1]
        acc = torch.float32 if Psi.dtype in (torch.bfloat16, torch.float16) else Psi.dtype   # bf16 atomics are ~10x slower
        F = Psi.new_zeros(B, N0, r, p, dtype=acc)
        Ff = F.view(B * N0, r * p)
        base = (torch.arange(B, device=Psi.device) * N0).view(B, 1, 1)
        for s in range(0, n, CHUNK):
            e = min(n + 0, s + CHUNK)
            for k in range(K):                                                   # weight folded into Psi: one (B,C,rp) write per pair
                op = torch.einsum("bjr,bjp->bjrp", Psi[:, s:e] * w[:, s:e, k:k + 1], Vb[:, s:e]).reshape(-1, r * p)
                Ff.index_add_(0, (idx[:, s:e, k] + base[:, :, 0]).reshape(-1), op.to(acc))
        ctx.save_for_backward(Psi, Vb, idx, w); ctx.N0 = N0
        return F.to(Psi.dtype)

    @staticmethod
    def backward(ctx, Fbar):
        Psi, Vb, idx, w = ctx.saved_tensors; N0 = ctx.N0
        B, n, r = Psi.shape; p = Vb.shape[-1]; K = idx.shape[-1]
        Fbf = Fbar.reshape(B * N0, r * p)
        dPsi = torch.zeros_like(Psi); dVb = torch.zeros_like(Vb); dw = torch.zeros_like(w)
        base = (torch.arange(B, device=Psi.device) * N0).view(B, 1)
        for s in range(0, n, CHUNK):
            e = min(n, s + CHUNK)
            for k in range(K):
                G = Fbf.index_select(0, (idx[:, s:e, k] + base).reshape(-1)).view(B, e - s, r, p)   # gathered F-bar
                wk = w[:, s:e, k:k + 1]
                GV = torch.einsum("bjrp,bjp->bjr", G, Vb[:, s:e])                 # G read once for dPsi and dw
                dPsi[:, s:e] += wk * GV
                dVb[:, s:e] += wk * torch.einsum("bjrp,bjr->bjp", G, Psi[:, s:e])
                dw[:, s:e, k] = (GV * Psi[:, s:e]).sum(-1)
        return dPsi, dVb, None, dw, None


class _Read(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Phi, idx, w, F):
        B, m, r = Phi.shape; N0, p = F.shape[1], F.shape[-1]; K = idx.shape[-1]
        y = Phi.new_zeros(B, m, p); Ff = F.reshape(B * N0, r * p)
        base = (torch.arange(B, device=Phi.device) * N0).view(B, 1)
        for s in range(0, m, CHUNK):
            e = min(m, s + CHUNK)
            for k in range(K):
                G = Ff.index_select(0, (idx[:, s:e, k] + base).reshape(-1)).view(B, e - s, r, p)
                y[:, s:e] += torch.einsum("bir,birp->bip", Phi[:, s:e] * w[:, s:e, k:k + 1], G)
        ctx.save_for_backward(Phi, idx, w, F)
        return y

    @staticmethod
    def backward(ctx, ybar):
        Phi, idx, w, F = ctx.saved_tensors
        B, m, r = Phi.shape; N0, p = F.shape[1], F.shape[-1]; K = idx.shape[-1]
        acc = torch.float32 if F.dtype in (torch.bfloat16, torch.float16) else F.dtype
        dPhi = torch.zeros_like(Phi); dw = torch.zeros_like(w); dF = torch.zeros_like(F, dtype=acc)
        dFf = dF.view(B * N0, r * p); bi = torch.arange(B, device=Phi.device).view(B, 1)
        base = (torch.arange(B, device=Phi.device) * N0).view(B, 1)
        Ff = F.reshape(B * N0, r * p)
        for s in range(0, m, CHUNK):
            e = min(m, s + CHUNK)
            for k in range(K):
                G = Ff.index_select(0, (idx[:, s:e, k] + base).reshape(-1)).view(B, e - s, r, p)
                wk = w[:, s:e, k:k + 1]
                Gy = torch.einsum("birp,bip->bir", G, ybar[:, s:e])                 # G read once for dPhi and dw
                dPhi[:, s:e] += wk * Gy
                dw[:, s:e, k] = (Gy * Phi[:, s:e]).sum(-1)
                op = torch.einsum("bir,bip->birp", Phi[:, s:e] * wk, ybar[:, s:e]).reshape(-1, r * p)
                dFf.index_add_(0, (idx[:, s:e, k] + base).reshape(-1), op.to(acc))
        return dPhi, None, dw, dF.to(F.dtype)


def pool_segmented(Psi, Vb, idx, w, N0):
    return _Pool.apply(Psi.contiguous(), Vb.contiguous(), idx.contiguous(), w.contiguous(), N0)


def read_gather(Phi, idx, w, F):
    return _Read.apply(Phi.contiguous(), idx.contiguous(), w.contiguous(), F.contiguous())


if __name__ == "__main__":
    # exactness vs the dense formulation, in float64, with gradients
    torch.manual_seed(0)
    B, n, m, r, p, N0, K = 3, 300, 200, 5, 4, 7, 4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Psi = torch.randn(B, n, r, device=dev, dtype=torch.float64, requires_grad=True)
    Vb = torch.randn(B, n, p, device=dev, dtype=torch.float64, requires_grad=True)
    Phi = torch.randn(B, m, r, device=dev, dtype=torch.float64, requires_grad=True)
    idk = torch.randint(0, N0, (B, n, K), device=dev); wk = torch.rand(B, n, K, device=dev, dtype=torch.float64, requires_grad=True)
    idq = torch.randint(0, N0, (B, m, K), device=dev); wq = torch.rand(B, m, K, device=dev, dtype=torch.float64, requires_grad=True)
    # dense reference
    Wk = torch.zeros(B, n, N0, device=dev, dtype=torch.float64).scatter_add(-1, idk, wk)
    Wq = torch.zeros(B, m, N0, device=dev, dtype=torch.float64).scatter_add(-1, idq, wq)
    Fd = torch.einsum("bnx,bnr,bnp->bxrp", Wk, Psi, Vb)
    yd = torch.einsum("bix,bir,bxrp->bip", Wq, Phi, Fd)
    gd = torch.autograd.grad(yd.sum(), (Psi, Vb, Phi, wk, wq))
    # segmented
    F = pool_segmented(Psi, Vb, idk, wk, N0); y = read_gather(Phi, idq, wq, F)
    gs = torch.autograd.grad(y.sum(), (Psi, Vb, Phi, wk, wq))
    print("forward max|err|", (y - yd).abs().max().item())
    for name, a, b in zip(("dPsi", "dVb", "dPhi", "dwk", "dwq"), gs, gd):
        print(f"{name} max|err| {(a - b).abs().max().item():.2e}")
    assert (y - yd).abs().max() < 1e-10 and all((a - b).abs().max() < 1e-9 for a, b in zip(gs, gd)); print("OK")


# ---------------------------------------------------------------------------------------------------------------
# Sorted, capped-padded formulation: group (token, k) pairs by cell, split each cell into sub-rows of LCAP slots, and
# do the pooling and the read as ONE batched matmul each over the sub-rows (plus one small index_add of the per-sub-row
# partial sums).  Plain torch ops, so autograd handles the backward; memory is O((pairs + B*N0*LCAP) * (r + p)) for
# any hash skew, and no per-token atomics.  Exact.
# ---------------------------------------------------------------------------------------------------------------
def _group(idx, w, N0):
    """idx, w: (B, L, K) -> per (b, cell) padded slot layout.  Returns (slot tensor of shape (B*L*K,) giving the
    flat index into (B, N0, Lmax), Lmax, and the flat (b, cell) order)."""
    B, L, K = idx.shape
    flat_cell = (idx + (torch.arange(B, device=idx.device) * N0).view(B, 1, 1)).reshape(-1)     # (B*L*K,)
    order = torch.argsort(flat_cell, stable=True)
    sorted_cell = flat_cell[order]
    counts = torch.bincount(sorted_cell, minlength=B * N0)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(sorted_cell.numel(), device=idx.device) - starts[sorted_cell]              # position within cell
    Lmax = int(counts.max().item())
    LAST["Lmax"], LAST["rows"], LAST["mean"] = Lmax, B * N0 * Lmax, L * K / N0
    slot = torch.empty_like(order); slot[order] = sorted_cell * Lmax + pos                        # flat index into (B*N0, Lmax)
    return slot, Lmax


import os
LCAP = int(os.environ.get("SMAT_LCAP", 64))   # sub-row length: hot cells are split into ceil(count / LCAP) sub-rows
LAST = {}                                      # diagnostics from the last _group call


def _group(idx, w, N0, Lcap=None):
    """idx: (B, L, K) cells -> capped padded layout.  Pairs are sorted by (b, cell); every cell is split into
    ceil(count / Lcap) sub-rows of Lcap slots, so the padded tensors are O(pairs + B*N0*Lcap) whatever the skew
    (a hot cell costs count/Lcap sub-rows, not a global Lmax), and the only atomics are one index_add of the
    per-sub-row partial sums.  Returns (slot: flat index of each (b, token, k) pair into (n_sub * Lcap,),
    n_sub, cell_of_sub: (n_sub,) flat (b, cell) index of each sub-row)."""
    Lcap = Lcap or LCAP
    B, L, K = idx.shape
    flat_cell = (idx + (torch.arange(B, device=idx.device) * N0).view(B, 1, 1)).reshape(-1)     # (B*L*K,)
    order = torch.argsort(flat_cell, stable=True)
    sorted_cell = flat_cell[order]
    counts = torch.bincount(sorted_cell, minlength=B * N0)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(sorted_cell.numel(), device=idx.device) - starts[sorted_cell]              # position within cell
    nsub = (counts + Lcap - 1) // Lcap                                                           # sub-rows per cell
    sub_base = torch.cumsum(nsub, 0) - nsub
    n_sub = int(nsub.sum().item())
    row = sub_base[sorted_cell] + pos // Lcap
    slot = torch.empty_like(order); slot[order] = row * Lcap + pos % Lcap
    cell_of_sub = torch.repeat_interleave(torch.arange(B * N0, device=idx.device), nsub, output_size=n_sub)
    LAST["Lmax"], LAST["n_sub"], LAST["rows"], LAST["mean"] = int(counts.max().item()), n_sub, n_sub * Lcap, L * K / N0
    return slot, n_sub, cell_of_sub


def pool_sorted(Psi, Vb, idx, w, N0):
    B, n, r = Psi.shape; p = Vb.shape[-1]; K = idx.shape[-1]
    slot, n_sub, cell_of_sub = _group(idx, w, N0); Lcap = LCAP
    Pw = (Psi.unsqueeze(2) * w.unsqueeze(-1)).reshape(B * n * K, r)                              # weight folded into Psi
    Vk = Vb.reshape(B * n, p) if K == 1 else Vb.unsqueeze(2).expand(B, n, K, p).reshape(B * n * K, p)
    Pp = Psi.new_zeros(n_sub * Lcap, r).index_copy(0, slot, Pw).view(n_sub, Lcap, r)
    Vp = Vb.new_zeros(n_sub * Lcap, p).index_copy(0, slot, Vk).view(n_sub, Lcap, p)
    Fsub = torch.bmm(Pp.transpose(1, 2), Vp).reshape(n_sub, r * p)                               # per-sub-row partial sums
    acc = torch.float32 if Fsub.dtype in (torch.bfloat16, torch.float16) else Fsub.dtype
    F = Fsub.new_zeros(B * N0, r * p, dtype=acc).index_add(0, cell_of_sub, Fsub.to(acc))
    return F.to(Psi.dtype).view(B, N0, r, p)                                                     # (B, N0, r, p)


def read_sorted(Phi, idx, w, F):
    B, m, r = Phi.shape; N0, p = F.shape[1], F.shape[-1]; K = idx.shape[-1]
    slot, n_sub, cell_of_sub = _group(idx, w, N0); Lcap = LCAP
    Qw = (Phi.unsqueeze(2) * w.unsqueeze(-1)).reshape(B * m * K, r)
    Qp = Phi.new_zeros(n_sub * Lcap, r).index_copy(0, slot, Qw).view(n_sub, Lcap, r)
    Fs = F.reshape(B * N0, r, p).index_select(0, cell_of_sub)                                    # (n_sub, r, p)
    Yp = torch.bmm(Qp, Fs).view(n_sub * Lcap, p)
    Yk = Yp[slot].view(B, m, K, p)
    return Yk.squeeze(2) if K == 1 else Yk.sum(2)


if __name__ == "__main__":
    for LCAP in (64, 3, 1):      # normal, and small caps that force hot cells to split into many sub-rows
        F2 = pool_sorted(Psi, Vb, idk, wk, N0); y2 = read_sorted(Phi, idq, wq, F2)
        g2 = torch.autograd.grad(y2.sum(), (Psi, Vb, Phi, wk, wq))
        print(f"capped-padded (LCAP={LCAP}) n_sub={LAST['n_sub']} forward max|err|", (y2 - yd).abs().max().item(), " grads", max((a - b).abs().max().item() for a, b in zip(g2, gd)))
        assert (y2 - yd).abs().max() < 1e-10 and all((a - b).abs().max() < 1e-9 for a, b in zip(g2, gd)); print("SORTED OK")

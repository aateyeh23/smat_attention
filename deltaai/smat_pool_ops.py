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
        F = Psi.new_zeros(B, N0, r, p)
        Ff = F.view(B * N0, r * p)
        base = (torch.arange(B, device=Psi.device) * N0).view(B, 1, 1)
        for s in range(0, n, CHUNK):
            e = min(n + 0, s + CHUNK)
            op = torch.einsum("bjr,bjp->bjrp", Psi[:, s:e], Vb[:, s:e]).reshape(B, e - s, r * p)   # (B, C, rp)
            for k in range(K):
                Ff.index_add_(0, (idx[:, s:e, k] + base[:, :, 0]).reshape(-1), (op * w[:, s:e, k:k + 1]).reshape(-1, r * p))
        ctx.save_for_backward(Psi, Vb, idx, w); ctx.N0 = N0
        return F

    @staticmethod
    def backward(ctx, Fbar):
        Psi, Vb, idx, w = ctx.saved_tensors; N0 = ctx.N0
        B, n, r = Psi.shape; p = Vb.shape[-1]; K = idx.shape[-1]
        Fb = Fbar.reshape(B, N0, r, p)
        dPsi = torch.zeros_like(Psi); dVb = torch.zeros_like(Vb); dw = torch.zeros_like(w)
        bi = torch.arange(B, device=Psi.device).view(B, 1)
        for s in range(0, n, CHUNK):
            e = min(n, s + CHUNK)
            for k in range(K):
                G = Fb[bi, idx[:, s:e, k]]                                   # (B, C, r, p) gathered F-bar
                wk = w[:, s:e, k:k + 1]
                dPsi[:, s:e] += wk * torch.einsum("bjrp,bjp->bjr", G, Vb[:, s:e])
                dVb[:, s:e] += wk * torch.einsum("bjrp,bjr->bjp", G, Psi[:, s:e])
                dw[:, s:e, k] = torch.einsum("bjr,bjrp,bjp->bj", Psi[:, s:e], G, Vb[:, s:e])
        return dPsi, dVb, None, dw, None


class _Read(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Phi, idx, w, F):
        B, m, r = Phi.shape; N0, p = F.shape[1], F.shape[-1]; K = idx.shape[-1]
        y = Phi.new_zeros(B, m, p); bi = torch.arange(B, device=Phi.device).view(B, 1)
        for s in range(0, m, CHUNK):
            e = min(m, s + CHUNK)
            for k in range(K):
                G = F[bi, idx[:, s:e, k]]                                    # (B, C, r, p)
                y[:, s:e] += w[:, s:e, k:k + 1] * torch.einsum("bir,birp->bip", Phi[:, s:e], G)
        ctx.save_for_backward(Phi, idx, w, F)
        return y

    @staticmethod
    def backward(ctx, ybar):
        Phi, idx, w, F = ctx.saved_tensors
        B, m, r = Phi.shape; N0, p = F.shape[1], F.shape[-1]; K = idx.shape[-1]
        dPhi = torch.zeros_like(Phi); dw = torch.zeros_like(w); dF = torch.zeros_like(F)
        dFf = dF.view(B * N0, r * p); bi = torch.arange(B, device=Phi.device).view(B, 1)
        base = (torch.arange(B, device=Phi.device) * N0).view(B, 1)
        for s in range(0, m, CHUNK):
            e = min(m, s + CHUNK)
            op = torch.einsum("bir,bip->birp", Phi[:, s:e], ybar[:, s:e]).reshape(B, e - s, r * p)
            for k in range(K):
                G = F[bi, idx[:, s:e, k]]
                wk = w[:, s:e, k:k + 1]
                dPhi[:, s:e] += wk * torch.einsum("birp,bip->bir", G, ybar[:, s:e])
                dw[:, s:e, k] = torch.einsum("bir,birp,bip->bi", Phi[:, s:e], G, ybar[:, s:e])
                dFf.index_add_(0, (idx[:, s:e, k] + base).reshape(-1), (op * wk).reshape(-1, r * p))
        return dPhi, None, dw, dF


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
# Sorted-and-padded formulation: group (token, k) pairs by cell, pad each cell to the max occupancy, and do the
# pooling and the read as ONE batched matmul each.  Plain torch ops, so autograd handles the backward; memory is
# O(B * N0 * Lmax * (r + p)), a few hundred MB at LM scale when the hash is balanced.  Exact.
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
    slot = torch.empty_like(order); slot[order] = sorted_cell * Lmax + pos                        # flat index into (B*N0, Lmax)
    return slot, Lmax


def pool_sorted(Psi, Vb, idx, w, N0):
    B, n, r = Psi.shape; p = Vb.shape[-1]; K = idx.shape[-1]
    slot, Lmax = _group(idx, w, N0)
    Pw = (Psi.unsqueeze(2) * w.unsqueeze(-1)).reshape(B * n * K, r)                              # weight folded into Psi
    Vk = Vb.unsqueeze(2).expand(B, n, K, p).reshape(B * n * K, p)
    Pp = Psi.new_zeros(B * N0 * Lmax, r).index_copy(0, slot, Pw).view(B * N0, Lmax, r)
    Vp = Vb.new_zeros(B * N0 * Lmax, p).index_copy(0, slot, Vk).view(B * N0, Lmax, p)
    return torch.bmm(Pp.transpose(1, 2), Vp).view(B, N0, r, p)                                   # (B, N0, r, p)


def read_sorted(Phi, idx, w, F):
    B, m, r = Phi.shape; N0, p = F.shape[1], F.shape[-1]; K = idx.shape[-1]
    slot, Lmax = _group(idx, w, N0)
    Qw = (Phi.unsqueeze(2) * w.unsqueeze(-1)).reshape(B * m * K, r)
    Qp = Phi.new_zeros(B * N0 * Lmax, r).index_copy(0, slot, Qw).view(B * N0, Lmax, r)
    Yp = torch.bmm(Qp, F.reshape(B * N0, r, p)).view(B * N0 * Lmax, p)                            # (B*N0*Lmax, p)
    Yk = Yp[slot].view(B, m, K, p)
    return Yk.sum(2)


if __name__ == "__main__":
    F2 = pool_sorted(Psi, Vb, idk, wk, N0); y2 = read_sorted(Phi, idq, wq, F2)
    g2 = torch.autograd.grad(y2.sum(), (Psi, Vb, Phi, wk, wq))
    print("sorted forward max|err|", (y2 - yd).abs().max().item(), " grads", max((a - b).abs().max().item() for a, b in zip(g2, gd)))
    assert (y2 - yd).abs().max() < 1e-10 and all((a - b).abs().max() < 1e-9 for a, b in zip(g2, gd)); print("SORTED OK")

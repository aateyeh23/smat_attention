"""Prefill cost under a content-addressed assignment: the dense reference against
the schedule Theorem "exact" actually describes.

The accuracy runs pool by profile with a dense contraction -- one pass over the
whole distant block per cell,

    F_x = sum_j  1{prof(j) = x} psi_j vb_j^T        for x = 0 .. N0-1

written as ``torch.stack([(Psi * W[:, :, x:x+1]).T @ Vb for x in range(N0)])``,
which costs Theta(N0 n r p) and not the O(n r p) of a grouped reduction.  The
read side is the same story: the query pass builds every (direction, offset)
state for every recent row and then selects one, Theta(T_R B r p) where
Theta(T_R r p) suffices.  The forward weights are one-hot, so the dense route
agrees with the grouped one in value and every published accuracy number is the
one the fast schedule would produce -- but the timing claim has never been made,
and the draft says so in red.

This measures it.  Three implementations of each half, checked against the dense
reference for exactness and then timed:

  pool   dense   the reference above
         scatter one pass, ``index_add_`` by profile -- the scatter the note asks
                 for; chunked over the sequence so memory stays bounded
  read   dense   the reference: build all B states per row, then select
         gather  one state per query, contracted directly
         sort    counting sort of queries by type, then one GEMM per type over
                 its contiguous block -- no per-query state is ever materialised

Shapes are drawn rather than produced by a model: the cost of either schedule
depends on the cell histogram and the tensor sizes, not on what the hash learned,
and cells drawn uniformly are what a balanced hash gives (the spread hypothesis
of the recall proposition, and what the id table achieves exactly).
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))          # the smat package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling task modules
from smat.mask import build_mask                                   # noqa: E402
from smat.assign import _grouped_planes                           # noqa: E402


# ---------------------------------------------------------------------------
# pooling by profile
# ---------------------------------------------------------------------------
def pool_dense(Psi, Vb, cells, N0):
    """The reference: one full pass over the distant block per cell."""
    W = torch.zeros(Psi.shape[0], Psi.shape[1], N0, device=Psi.device, dtype=Psi.dtype)
    W.scatter_(2, cells.unsqueeze(-1), 1.0)
    return torch.stack([(Psi * W[:, :, x:x + 1]).transpose(-1, -2) @ Vb
                        for x in range(N0)], dim=1)


def pool_scatter(Psi, Vb, cells, N0, chunk=8192):
    """Grouped reduction: every key is written once, into its own cell.

    ``prof`` is a function, so the selector has exactly one nonzero per row and
    the pool is a scatter-add rather than a matrix product.  That is the whole
    content of the O(n r p) claim.  The chunk bounds the outer products held at
    once; it does not change the work.
    """
    nbh, n, r = Psi.shape
    p = Vb.shape[-1]
    F = torch.zeros(nbh * N0, r, p, device=Psi.device, dtype=torch.float32)
    base = (torch.arange(nbh, device=Psi.device) * N0).view(nbh, 1)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        outer = Psi[:, s:e].unsqueeze(-1) * Vb[:, s:e].unsqueeze(-2)   # (nbh, c, r, p)
        idx = (base + cells[:, s:e]).reshape(-1)
        F.index_add_(0, idx, outer.reshape(-1, r, p))
    return F.view(nbh, N0, r, p)


# ---------------------------------------------------------------------------
# reading by type
# ---------------------------------------------------------------------------
def read_dense(Phi_r, U, types):
    """The reference: contract every row against every type, then select."""
    Z = torch.einsum("bir,btrp->bitp", Phi_r, U)                   # (nbh, TR, B, p)
    return Z.gather(2, types.view(*types.shape, 1, 1)
                    .expand(*types.shape, 1, Z.shape[-1])).squeeze(2)


def read_gather(Phi_r, U, types, chunk=8192):
    """One state per query, contracted directly.  O(T_R r p) work."""
    nbh, TR, r = Phi_r.shape
    p = U.shape[-1]
    out = torch.empty(nbh, TR, p, device=Phi_r.device, dtype=torch.float32)
    for s in range(0, TR, chunk):
        e = min(s + chunk, TR)
        idx = types[:, s:e].view(nbh, e - s, 1, 1).expand(nbh, e - s, r, p)
        Ug = U.gather(1, idx)                                      # (nbh, c, r, p)
        out[:, s:e] = (Phi_r[:, s:e].unsqueeze(-2) @ Ug).squeeze(-2)
    return out


def read_sort(Phi_r, U, types):
    """Counting sort of queries by type, then one GEMM per type.

    Rows sharing a type read the same state, so sorting makes each type's rows
    contiguous and the pass becomes one dense product per type with no
    per-query state materialised at all.  This is the read side of the note.
    """
    nbh, TR, r = Phi_r.shape
    B, p = U.shape[1], U.shape[-1]
    out = torch.empty(nbh, TR, p, device=Phi_r.device, dtype=torch.float32)
    order = torch.argsort(types, dim=1)                            # counting sort
    st = types.gather(1, order)
    # boundaries of each type's block, per batch row
    bounds = torch.searchsorted(st, torch.arange(B + 1, device=st.device)
                                .unsqueeze(0).expand(nbh, B + 1).contiguous())
    for b in range(nbh):
        ob, Ub = order[b], U[b]
        for t in range(B):
            lo, hi = int(bounds[b, t]), int(bounds[b, t + 1])
            if hi > lo:
                rows = ob[lo:hi]
                out[b, rows] = Phi_r[b, rows] @ Ub[t]
    return out


# ---------------------------------------------------------------------------
def bench(fn, device, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / repeat


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--nb", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--dv", type=int, default=64)
    ap.add_argument("--dense-max-T", type=int, default=16384,
                    help="the reference is quadratic in the wrong place; do not "
                         "run it past this or the job is all reference")
    ap.add_argument("--sort-max-B", type=int, default=64,
                    help="the counting-sort read loops in python over nbh x B; "
                         "past this it is launch-bound and not informative")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default="results/bench_ca.csv")
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(json.dumps({**vars(args), "device": dev}), flush=True)

    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.DictWriter(f, ["d", "T", "n", "N0", "B", "nbh", "r", "p",
                           "pool_dense_ms", "pool_scatter_ms", "pool_speedup",
                           "read_dense_ms", "read_gather_ms", "read_sort_ms",
                           "read_speedup", "pool_max_err", "read_max_err"])
    if new:
        w.writeheader()

    nbh, r, p = args.nb * args.heads, args.r, args.dv + 1
    for T in args.T:
        try:
            spec = build_mask(T, args.d, chunk=args.chunk)
            n, N0 = spec.n, int(spec.N0)
            dirs, _ = _grouped_planes(int(spec.q), int(spec.dim))
            B = int(dirs.shape[0]) * int(spec.q)
            TR = T - n
            Psi = torch.rand(nbh, n, r, device=dev)
            Vb = torch.randn(nbh, n, p, device=dev)
            Phi = torch.rand(nbh, TR, r, device=dev)
            cells = torch.randint(0, N0, (nbh, n), device=dev)
            types = torch.randint(0, B, (nbh, TR), device=dev)
            row = {"d": args.d, "T": T, "n": n, "N0": N0, "B": B,
                   "nbh": nbh, "r": r, "p": p}

            Us = pool_scatter(Psi, Vb, cells, N0)
            row["pool_scatter_ms"] = bench(lambda: pool_scatter(Psi, Vb, cells, N0),
                                           dev, 1, args.repeat)
            if T <= args.dense_max_T:
                Ud = pool_dense(Psi, Vb, cells, N0)
                row["pool_max_err"] = (Ud - Us).abs().max().item()
                row["pool_dense_ms"] = bench(lambda: pool_dense(Psi, Vb, cells, N0),
                                             dev, 1, args.repeat)
                row["pool_speedup"] = row["pool_dense_ms"] / row["pool_scatter_ms"]
                del Ud
            else:
                row["pool_dense_ms"] = row["pool_max_err"] = row["pool_speedup"] = ""

            # the type table the read works from, shaped (nbh, B, r, p)
            U = torch.randn(nbh, B, r, p, device=dev)
            Rg = read_gather(Phi, U, types)
            row["read_gather_ms"] = bench(lambda: read_gather(Phi, U, types),
                                          dev, 1, args.repeat)
            if B <= args.sort_max_B:
                Rs = read_sort(Phi, U, types)
                row["read_sort_ms"] = bench(lambda: read_sort(Phi, U, types),
                                            dev, 1, args.repeat)
                del Rs
            else:
                row["read_sort_ms"] = ""
            if T <= args.dense_max_T:
                Rd = read_dense(Phi, U, types)
                row["read_max_err"] = (Rd - Rg).abs().max().item()
                row["read_dense_ms"] = bench(lambda: read_dense(Phi, U, types),
                                             dev, 1, args.repeat)
                row["read_speedup"] = row["read_dense_ms"] / row["read_gather_ms"]
                del Rd
            else:
                row["read_dense_ms"] = row["read_max_err"] = row["read_speedup"] = ""

            pd = row["pool_dense_ms"]
            rd = row["read_dense_ms"]
            print(f"  T={T:>7} n={n:>7} N0={N0:>4} B={B:>5}  "
                  f"pool {('%9.2f' % pd) if pd != '' else '        -'} -> "
                  f"{row['pool_scatter_ms']:7.2f} ms  "
                  f"read {('%9.2f' % rd) if rd != '' else '        -'} -> "
                  f"{row['read_gather_ms']:7.2f} ms  "
                  f"err {row['pool_max_err'] if row['pool_max_err'] == '' else '%.1e' % row['pool_max_err']}"
                  f" / {row['read_max_err'] if row['read_max_err'] == '' else '%.1e' % row['read_max_err']}",
                  flush=True)
            w.writerow(row); f.flush()
            del Psi, Vb, Phi, U, Us, Rg
            if dev == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  T={T}: FAILED {type(e).__name__}: {e}", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()
    f.close()


if __name__ == "__main__":
    main()

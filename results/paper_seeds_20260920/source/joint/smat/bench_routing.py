"""d-Subset Routing: the expressivity benchmark of Sec. 4.

The lemma of "Notions of Expressiveness" makes this task measurable without
training.  Mark ``k`` context positions ``J = {j_1..j_k}``, put ``v_{j_l} =
x_l e_l`` and zero every other value.  Then for *any* attention kernel strictly
positive on allowed edges,

    supp(o_t) = S_t ∩ J                                          (the lemma)

so the set of routing patterns the layer can produce on the marked channels is
exactly ``{S_t ∩ J : t in [T]}``, whose cardinality is the shatter function
``Pi_M(k)``.  A request ``A ⊆ J`` is *answerable* iff some row realises it, and
``R(M) = VC(M)`` is the largest ``k`` for which every request is answerable.

That makes the "controlled transition as the requested routing dimension
crosses VC(M)" a property of the mask that can be certified rather than
estimated: no optimiser, no seeds, no confound between what the architecture
permits and what SGD happened to find.  What is measured here is therefore the
*ceiling* for any model carrying this mask.  A learned run would show a trained
model approaching it; it cannot exceed it.

Three levels, in increasing strength:

  shatter    Pi_M(k) counted exactly, against 2^k and Sauer--Shelah
  answerable the fraction of requests A ⊆ J that some row realises
  task       the task of Sec. 4 run end to end through the real SMAT kernel,
             scoring ||yhat - y_A|| against the stated target

Level 3 is the one that can fail for reasons levels 1--2 cannot see: it uses
``smat_attention`` itself, so a kernel bug, a normalisation error or a
fp-accumulation problem shows up as task error even when the combinatorics are
right.

Usage
-----
    python bench_routing.py --T 4096 --kmax 6                 # ~1 min, cpu
    python bench_routing.py --T 8192 --d 1 2 3 4 --plot
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from smat_mask import (MaskSpec, build_mask, d_max_geometric, shatter_witness)

__all__ = ["mask_columns", "row_support_size", "pattern_table",
           "shatter_profile", "choose_marked_set", "run_task",
           "sauer_shelah"]


# ---------------------------------------------------------------------------
# row supports on a marked set, without materialising M
# ---------------------------------------------------------------------------

def mask_columns(spec: MaskSpec, cols: Sequence[int]) -> np.ndarray:
    """``M[:, cols]`` as a ``(T, len(cols))`` bool array, in ``O(T |cols|)``.

    The dense route is ``O(T^2)``; this reads the same entries off the
    definition, so the benchmark runs at the ``T`` of the timing sweep.
    """
    T, n, n_P, B = spec.T, spec.n, spec.n_P, spec.B
    cols = np.asarray(cols, dtype=np.int64)
    out = np.zeros((T, cols.size), dtype=bool)

    C = spec.dense_C().astype(bool)
    prof = spec.prof_of_column()
    rows_rec = np.arange(spec.T_R, dtype=np.int64)
    rho = rows_rec % B

    land = np.arange(n, dtype=np.int64)
    for a, j in enumerate(cols):
        if j < n:                                   # a landmark column
            # top block: L_n (geometric) or Z_k (endpoint)
            if spec.kind == "endpoint":
                out[:n, a] = (land & j) == j
            else:
                out[:n, a] = land >= j
            # bottom block: the global channel P, else G = R C S^T
            if j < n_P:
                out[n:, a] = True
            else:
                out[n:, a] = C[rho, prof[j - n_P]]
        else:                                       # a recent column
            out[n:, a] = rows_rec >= (j - n)        # L_{T_R}; top block is 0
    return out


def row_support_size(spec: MaskSpec) -> np.ndarray:
    """``|S_t|`` for every row, shape ``(T,)`` -- the attention normalizer.

    Needed to undo the mean that attention takes: with a kernel constant on
    allowed edges the layer returns ``y_A / |S_t|``, so the task readout
    rescales by this known, mask-determined quantity.
    """
    T, n, n_P, B = spec.T, spec.n, spec.n_P, spec.B
    out = np.zeros(T, dtype=np.int64)

    land = np.arange(n, dtype=np.int64)
    if spec.kind == "endpoint":
        out[:n] = 1 << np.array([int(x).bit_count() for x in land])
    else:
        out[:n] = land + 1                                   # L_n

    # landmarks per profile, then per row type
    n_X, N0 = spec.n_X, spec.N0
    per_prof = np.full(N0, n_X // N0, dtype=np.int64)
    per_prof[:n_X % N0] += 1
    C = spec.dense_C().astype(bool)
    per_type = C @ per_prof                                  # (B,)

    rows_rec = np.arange(spec.T_R, dtype=np.int64)
    out[n:] = n_P + per_type[rows_rec % B] + (rows_rec + 1)  # P + G + L_{T_R}
    return out


def pattern_table(spec: MaskSpec, cols: Sequence[int]) -> Dict[int, int]:
    """``{pattern bitmask -> a witnessing row}`` over all ``T`` rows.

    The pattern of row ``t`` is ``S_t ∩ J`` packed into an int, bit ``l`` set
    iff ``cols[l] in S_t``.
    """
    sup = mask_columns(spec, cols)
    weights = (1 << np.arange(len(cols), dtype=np.int64))
    packed = sup.astype(np.int64) @ weights                  # (T,)
    seen: Dict[int, int] = {}
    # first occurrence wins; np.unique gives the first index per value
    vals, idx = np.unique(packed, return_index=True)
    for v, i in zip(vals.tolist(), idx.tolist()):
        seen[int(v)] = int(i)
    return seen


def sauer_shelah(k: int, d: int, T: int) -> int:
    """``min(T, sum_{i<=d} C(k,i))`` -- the ceiling a VC-``d`` mask obeys."""
    return int(min(T, sum(math.comb(k, i) for i in range(d + 1))))


# ---------------------------------------------------------------------------
# choosing the marked set
# ---------------------------------------------------------------------------

def choose_marked_set(spec: MaskSpec, k: int, *, rng: np.random.Generator,
                      candidates: int = 192) -> Tuple[List[int], str]:
    """``k`` marked positions, starting from the construction's own witness.

    For ``k <= d`` the witness is shattered by construction, so it is the right
    set to score.  For ``k > d`` no set is shattered (that is the content of
    ``VC = d``), and which set is *least* bad is a search; we extend the witness
    greedily, maximising the realised pattern count at each step over a random
    subsample of candidate columns.  Greedy over a subsample is a good-faith
    effort, not an optimum -- reported ``Pi_M(k)`` for ``k > d`` is a lower
    bound on what the best marked set would give.
    """
    wit, ok, _ = shatter_witness(spec)
    wit = [int(c) for c in wit]
    if k <= len(wit):
        return wit[:k], "witness" if ok else "witness(unverified)"

    cols = list(wit)
    pool = np.arange(spec.n_P, spec.n, dtype=np.int64)        # landmark columns
    while len(cols) < k:
        pool_free = np.setdiff1d(pool, np.array(cols, dtype=np.int64))
        if pool_free.size == 0:
            break
        pick = rng.choice(pool_free, size=min(candidates, pool_free.size),
                          replace=False)
        best, best_n = None, -1
        for c in pick.tolist():
            npat = len(pattern_table(spec, cols + [c]))
            if npat > best_n:
                best, best_n = c, npat
        cols.append(int(best))
    return cols, "witness+greedy"


def shatter_profile(spec: MaskSpec, cols: Sequence[int]) -> Dict[str, object]:
    """``Pi_M(k)`` on ``cols``, and the fraction of requests that are answerable."""
    k = len(cols)
    pat = pattern_table(spec, cols)
    answerable = sum(1 for A in range(1 << k) if A in pat)
    return dict(k=k, n_patterns=len(pat), n_requests=1 << k,
                answerable=answerable, answerable_frac=answerable / (1 << k),
                full_2k=1 << k, sauer=sauer_shelah(k, spec.d, spec.T))


# ---------------------------------------------------------------------------
# the task itself, through the real kernel
# ---------------------------------------------------------------------------

def _nearest_row(pat: Dict[int, int], A: int, k: int) -> int:
    """The row whose pattern is closest to ``A`` in Hamming distance.

    A model handed an unanswerable request must still answer; this is the best
    row available to it, and the resulting error is the irreducible one the
    mask imposes.
    """
    if A in pat:
        return pat[A]
    best, best_d = None, k + 1
    for p, t in pat.items():
        dist = int(p ^ A).bit_count()
        if dist < best_d:
            best, best_d = t, dist
    return best


def run_task(spec: MaskSpec, cols: Sequence[int], *, n_examples: int = 64,
             rng: np.random.Generator, device: str = "cpu",
             dtype: str = "fp32", requests: str = "all") -> Dict[str, object]:
    """Sec. 4's task end to end: payloads in, ``smat_attention``, score vs ``y_A``.

    ``v_{j_l} = x_l e_l`` with ``x_l`` resampled per example, every other value
    zero.  Keys and queries are constant, so the kernel is constant on allowed
    edges and the layer returns ``y_A / |S_t|``; the readout multiplies the
    known ``|S_t|`` back.  Scoring is exact-match on the support (the lemma)
    plus normalised squared error on the payloads (the stated target).
    """
    import torch
    from smat_attn import featurise, make_phi, smat_attention, to_device

    k = len(cols)
    td = dict(fp32=torch.float32, fp64=torch.float64,
              bf16=torch.bfloat16)[dtype]
    dev = torch.device(device)
    T = spec.T

    pat = pattern_table(spec, cols)
    nu = row_support_size(spec)

    # requests to score
    if requests == "all" or (1 << k) <= n_examples:
        As = list(range(1 << k))
    else:
        As = rng.choice(1 << k, size=n_examples, replace=False).tolist()

    # one head; the kernel must be constant on allowed edges, so Q = K = 1.
    r = 8
    phi = make_phi(4, r, device=dev, dtype=td, seed=0)
    Q = torch.ones(1, T, 4, device=dev, dtype=td)
    K = torch.ones(1, T, 4, device=dev, dtype=td)

    spec_d = to_device(spec, dev)

    rows, exact_supp, nsq, leak = [], [], [], []
    for A in As:
        x = rng.standard_normal(k)
        V = torch.zeros(1, T, k, device=dev, dtype=td)
        for l, j in enumerate(cols):
            V[0, int(j), l] = float(x[l])

        Phi, Psi, Vb = featurise(Q, K, V, phi)
        out = smat_attention(Phi, Psi, Vb, spec_d, chunk=128,
                             acc_dtype=torch.float32)

        t = _nearest_row(pat, A, k)
        yhat = out[0, t].double().cpu().numpy() * float(nu[t])
        y = np.array([x[l] if (A >> l) & 1 else 0.0 for l in range(k)])

        got = int(sum((1 << l) for l in range(k)
                      if abs(yhat[l]) > 1e-8 * max(1.0, abs(x[l]))))
        exact_supp.append(got == A)
        # normalise by the payload energy present, not by ||y_A||: the empty
        # request has y_A = 0, and it is precisely the request that tests
        # whether the layer correctly emits nothing.  train_routing.py scores
        # the learned run the same way so the two are comparable.
        denom = float(x @ x)
        nsq.append(float((yhat - y) @ (yhat - y)) / denom)
        off = [abs(yhat[l]) for l in range(k) if not (A >> l) & 1]
        leak.append(max(off) if off else 0.0)
        rows.append(t)

    return dict(n_scored=len(As),
                exact_support_frac=float(np.mean(exact_supp)),
                nmse=float(np.mean(nsq)),
                nmse_answerable=float(np.mean(
                    [v for A, v in zip(As, nsq) if A in pat]) if
                    any(A in pat for A in As) else float("nan")),
                max_leak=float(np.max(leak)) if leak else 0.0)


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def sweep(args) -> List[Dict]:
    rng = np.random.default_rng(args.seed)
    out: List[Dict] = []
    for d in args.d:
        if d > d_max_geometric(args.T, chunk=args.chunk):
            print(f"  d={d}: skipped (above d_max({args.T})="
                  f"{d_max_geometric(args.T, chunk=args.chunk)})")
            continue
        spec = build_mask(args.T, d, chunk=args.chunk, n_P=args.n_P)
        print(f"\n{spec.summary()}")
        wit, ok, _ = shatter_witness(spec)
        print(f"  witness {list(map(int, wit))} shattered={ok}")

        for k in range(1, args.kmax + 1):
            t0 = time.perf_counter()
            cols, how = choose_marked_set(spec, k, rng=rng,
                                          candidates=args.candidates)
            prof = shatter_profile(spec, cols)
            row = dict(T=args.T, d=d, kind=spec.kind, B=spec.B, source=how,
                       cols=" ".join(map(str, cols)), **prof)
            if not args.no_task:
                row.update(run_task(spec, cols, n_examples=args.examples,
                                    rng=rng, device=args.device,
                                    dtype=args.dtype))
            row["seconds"] = time.perf_counter() - t0
            out.append(row)
            flag = "" if prof["answerable_frac"] == 1.0 else "  <- not shattered"
            extra = (f" nmse {row['nmse']:.2e} supp {row['exact_support_frac']:.2f}"
                     if not args.no_task else "")
            print(f"  k={k}  Pi_M={prof['n_patterns']:>4} / 2^k="
                  f"{prof['full_2k']:>4} (Sauer {prof['sauer']:>4})  "
                  f"answerable {prof['answerable_frac']:6.3f}{extra}{flag}",
                  flush=True)
    return out


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path} ({len(rows)} rows)")


def plot(rows: List[Dict], outdir: str, dark: bool = False):
    """fig12, drawn by plot_smat so every panel shares one palette."""
    from plot_smat import apply_style, fig_routing, theme

    P = theme(dark)
    apply_style(P)
    os.makedirs(outdir, exist_ok=True)
    fig_routing(rows, P, outdir)
    print(f"figure in {outdir}/fig12_routing.pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--d", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--n-P", type=int, default=0)
    ap.add_argument("--examples", type=int, default=64,
                    help="requests scored when 2^k exceeds it")
    ap.add_argument("--candidates", type=int, default=192,
                    help="columns sampled per greedy extension step")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp64", "bf16"])
    ap.add_argument("--no-task", action="store_true",
                    help="combinatorics only; skip the kernel run")
    ap.add_argument("--out", default=None,
                    help="default: results/routing_T<T>.csv, which is the name "
                         "plot_smat.py looks for")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rows = sweep(args)
    write_csv(rows, args.out or f"results/routing_T{args.T}.csv")
    if args.plot:
        plot(rows, args.outdir, args.dark)

    print("\nsummary: largest k with every request answerable")
    for d in sorted({r["d"] for r in rows}):
        sub = [r for r in rows if r["d"] == d]
        full = [r["k"] for r in sub if r["answerable_frac"] == 1.0]
        R = max(full) if full else 0
        mark = "ok" if R == d else "MISMATCH"
        print(f"  d={d}: R(M)={R}   VC predicted {d}   [{mark}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

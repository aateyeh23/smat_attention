#!/usr/bin/env python3
"""
Exact VC dimension / Pollard pseudo-dimension of a (non-binary) matrix.

Definition
----------
Let ``M`` be an ``m x n`` real matrix.  Rows are the *functions* (hypotheses),
columns are the *domain points*.  A set of columns ``j_1 < ... < j_d`` is
*pseudo-shattered* if there exist thresholds ``t_1, ..., t_d`` such that every
one of the ``2^d`` sign patterns is realised by some row:

    for every  s in {0,1}^d   there is a row  i  with
        [ M[i, j_k] >= t_k ]  ==  s_k     for all k = 1..d

The pseudo-dimension ``P-dim(M)`` is the largest such ``d`` (0 if none).
For a 0/1 matrix this is exactly the VC dimension of the set system whose
incidence matrix is ``M``.

Algorithm
---------
Deciding the VC dimension is LOGNP-complete, so no polynomial algorithm is
expected.  This is an exact branch-and-bound search, engineered so that the
per-node cost is a handful of vectorised passes over the rows.

1. **Only ranks matter.**  Replace each entry by the rank of its value within
   its column.  A column with ``c`` distinct values offers exactly ``c - 1``
   candidate thresholds; threshold ``k`` is the predicate ``rank >= k``.

2. **Incremental partition refinement.**  A set of thresholded columns
   pseudo-shatters iff refining the trivial partition ``{all rows}`` by each of
   them leaves all ``2^d`` blocks non-empty.  The search adds one column at a
   time and abandons a branch the moment a block dies, so the ``2^d`` patterns
   are never enumerated explicitly.

3. **Nested-split filtering (the main speed-up).**  The splits of one column
   are nested, so the thresholds of column ``j`` that cut a block ``b`` are
   exactly those with ``min_rank(b) < k <= max_rank(b)`` -- a contiguous
   interval.  The thresholds that cut *every* current block are therefore the
   intersection of those intervals, obtained in one segmented min/max pass over
   the rows instead of testing each split against each block.

4. **Pruning.**
   * duplicate rows are collapsed (they can never separate a pattern), giving
     ``d <= floor(log2(#distinct rows))``;
   * columns are visited in a fixed order (the shattered set is a *set*), so
     only later columns stay in play;
   * a column whose interval is empty is dead for the whole subtree -- blocks
     only refine, so it is dropped once and never reconsidered.  The surviving
     count gives the sharp bound ``d <= depth + #live columns``;
   * a block of ``s`` rows survives at most ``floor(log2 s)`` further splits,
     so every child block must hold at least ``2^(best - depth)`` rows.

5. **Seeding.**  A randomised greedy pass first produces a good incumbent,
   which makes the size-based prune bite from the root.

Per node the work is ``O(m * #live columns)`` machine ops, and the depth never
exceeds ``log2(m)``.  The tree itself can still blow up on wide, high-cardinality
inputs -- pass ``time_limit=`` for a certified lower bound, or ``max_thresholds=``
to search a quantile subsample of thresholds (also a certified lower bound).

Usage
-----
    from vc_dimension import pseudo_dimension, pseudo_dimension_detailed

    d = pseudo_dimension(M)                       # rows = functions (default)
    d = pseudo_dimension(M, points="rows")        # transposed convention
    res = pseudo_dimension_detailed(M, time_limit=10.0)
    print(res.dimension, res.exact, res.witness)

CLI
---
    python vc_dimension.py data.csv [--points columns|rows] [--time-limit S]
    python vc_dimension.py --selftest
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "pseudo_dimension",
    "pseudo_dimension_detailed",
    "vc_dimension",
    "Witness",
    "Result",
]


# ---------------------------------------------------------------------------
# public result types
# ---------------------------------------------------------------------------

@dataclass
class Witness:
    """Certificate that ``columns`` are pseudo-shattered using ``thresholds``.

    ``rows[p]`` is a row index of the *original* matrix realising the pattern
    whose k-th bit ``(p >> k) & 1`` says whether
    ``M[rows[p], columns[k]] >= thresholds[k]``.
    """

    columns: List[int] = field(default_factory=list)
    thresholds: List[float] = field(default_factory=list)
    rows: List[int] = field(default_factory=list)

    @property
    def dimension(self) -> int:
        return len(self.columns)

    def verify(self, M, points: str = "columns") -> bool:
        """Independently re-check this witness against the matrix."""
        M = np.asarray(M)
        if points == "rows":
            M = M.T
        d = len(self.columns)
        if len(self.rows) != 1 << d or len(self.thresholds) != d:
            return False
        if len(set(self.columns)) != d:
            return False
        for p, i in enumerate(self.rows):
            for k, (j, t) in enumerate(zip(self.columns, self.thresholds)):
                if bool(M[i, j] >= t) != bool((p >> k) & 1):
                    return False
        return True


@dataclass
class Result:
    dimension: int
    exact: bool                 # False if a time/threshold limit cut the search
    upper_bound: int            # cheap bound: min(#usable columns, floor(log2 #rows))
    witness: Optional[Witness]
    nodes: int
    seconds: float

    def __str__(self) -> str:
        kind = "exact" if self.exact else "lower bound (search truncated)"
        return (f"pseudo-dimension = {self.dimension}  [{kind}]  "
                f"(upper bound {self.upper_bound}, {self.nodes} nodes, "
                f"{self.seconds:.3f}s)")


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def _rank_encode(Mu: np.ndarray, max_thresholds: Optional[int]):
    """Rank-encode the columns.

    Returns ``(cols, values, R)`` where ``cols[j]`` is the original index of
    usable column ``j``, ``values[j]`` holds its candidate thresholds indexed by
    rank, and ``R[i, j]`` is the rank of ``Mu[i, j]``.  Constant columns are
    dropped.  Threshold ``k`` (1 <= k < len(values[j])) is the predicate
    ``R[:, j] >= k``, i.e. ``Mu[:, j] >= values[j][k]``.

    With ``max_thresholds`` set, each column keeps only that many quantile-spaced
    thresholds -- the result is then a valid lower bound rather than exact.
    """
    cols: List[int] = []
    values: List[np.ndarray] = []
    ranks: List[np.ndarray] = []
    truncated = False

    for j in range(Mu.shape[1]):
        vals, inv = np.unique(Mu[:, j], return_inverse=True)
        if vals.size < 2:
            continue                                   # constant: no split
        if max_thresholds is not None and vals.size - 1 > max_thresholds:
            # Keep value 0 (the "below everything" level) plus a spread of cut
            # points, then re-rank rows into the coarsened level set.
            pick = np.unique(np.linspace(1, vals.size - 1, max_thresholds).astype(np.int64))
            keep = np.concatenate(([0], pick))
            inv = np.searchsorted(keep, inv, side="right") - 1
            vals = vals[keep]
            truncated = True
        cols.append(j)
        values.append(vals)
        ranks.append(inv.astype(np.int32, copy=False))

    if not cols:
        return [], [], np.zeros((Mu.shape[0], 0), np.int32), truncated
    R = np.ascontiguousarray(np.stack(ranks, axis=1))
    return cols, values, R, truncated


"""A *partition* is carried as ``(perm, offs, sizes)``: ``perm`` lists every row
once with the blocks laid out contiguously, ``offs`` gives each block's start and
``sizes`` its length.  Keeping the blocks contiguous means a refinement is one
stable boolean partition of ``perm`` instead of a Python loop over ``2^depth``
index arrays, so the cost per node is independent of the number of blocks."""


def _column_ranges(R: np.ndarray, part, cand: np.ndarray, need: int, span: int):
    """Per candidate column, the thresholds worth trying.

    Column ``cand[a]`` cuts block ``b`` iff its threshold ``k`` satisfies
    ``min_rank(b) < k <= max_rank(b)``, so intersecting over blocks leaves
    ``lo[a] < k <= hi[a]``; the column is dead (now and for the whole subtree)
    when ``hi[a] <= lo[a]``.

    Requiring at least ``need`` rows on each side of every block sharpens this.
    ``#{rows of b below k}`` is non-decreasing in ``k`` and ``#{at or above k}``
    is non-increasing, so the surviving thresholds again form an interval
    ``[k1[a], k2[a]]``, delimited by the ``need``-th order statistic of each
    block from either end.  Every ``k`` in that interval is guaranteed to yield
    a legal refinement, so the search never tries a split that fails.

    One segmented sort per node yields all four bounds: adding
    ``block_id * span`` before sorting keeps each block inside its own stretch
    of rows while ordering the ranks within it.
    """
    perm, offs, sizes = part
    nb = offs.size
    sub = R[np.ix_(perm, cand)]
    if nb > 1:
        key = np.repeat(np.arange(nb, dtype=sub.dtype), sizes)[:, None] * span
        if key.dtype == np.int32 and nb * span > np.iinfo(np.int32).max:
            sub, key = sub.astype(np.int64), key.astype(np.int64)
        sub = sub + key
        sub.sort(axis=0, kind="stable")     # radix sort: linear in the ranks
        sub -= key
    else:
        sub = np.sort(sub, axis=0, kind="stable")
    lo = sub[offs].max(0)                    # max over blocks of the min rank
    hi = sub[offs + sizes - 1].min(0)        # min over blocks of the max rank
    if need == 1:
        return lo, hi, lo + 1, hi
    k1 = sub[offs + need - 1].max(0) + 1
    k2 = sub[offs + sizes - need].min(0)
    return lo, hi, k1, k2


def _refine(colv: np.ndarray, part, k: int, need: int):
    """Split every block on ``rank >= k``, low halves first.

    Because a stable partition keeps the blocks in order, the new block index is
    ``old + n_blocks * bit``, i.e. the new feature lands in bit ``depth``.
    Returns ``None`` if any child is smaller than ``need`` and so cannot survive
    the splits still required.
    """
    perm, offs, sizes = part
    hit = colv[perm] >= k
    csum = np.empty(perm.size + 1, np.int64)
    csum[0] = 0
    np.cumsum(hit, dtype=np.int64, out=csum[1:])
    hcnt = csum[offs + sizes] - csum[offs]
    lcnt = sizes - hcnt
    if hcnt.min() < need or lcnt.min() < need:
        return None
    new_perm = np.concatenate((perm[~hit], perm[hit]))
    new_sizes = np.concatenate((lcnt, hcnt))
    new_offs = np.empty_like(new_sizes)
    new_offs[0] = 0
    np.cumsum(new_sizes[:-1], out=new_offs[1:])
    return new_perm, new_offs, new_sizes


def _trivial_partition(mu: int):
    return (np.arange(mu, dtype=np.int64),
            np.zeros(1, np.int64),
            np.full(1, mu, np.int64))


def _greedy_lower_bound(R, cols_R, mu, span, rng, restarts, sample=16):
    """Randomised greedy seed: repeatedly take the most balanced usable split.

    Returns ``(depth, [(column_position, threshold_rank), ...], partition)``.
    Only a lower bound, but cheap and usually tight.
    """
    ncols = R.shape[1]
    best_depth, best_choice, best_part = 0, [], _trivial_partition(mu)
    order = list(range(ncols))
    all_cols = np.arange(ncols)

    for _ in range(restarts):
        rng.shuffle(order)
        part = _trivial_partition(mu)
        choice: List[Tuple[int, int]] = []
        lo, hi, _, _ = _column_ranges(R, part, all_cols, 1, span)
        for jp in order:
            if hi[jp] <= lo[jp]:
                continue                               # dead column
            ks = np.arange(lo[jp] + 1, hi[jp] + 1)
            if ks.size > sample:                       # thin out huge columns
                ks = ks[np.linspace(0, ks.size - 1, sample).astype(np.int64)]
            pick, score = None, -1
            for k in ks:
                nxt = _refine(cols_R[jp], part, int(k), 1)
                s = int(nxt[2].min())
                if s > score:
                    score, pick = s, (int(k), nxt)
            k, part = pick
            choice.append((jp, k))
            lo, hi, _, _ = _column_ranges(R, part, all_cols, 1, span)
        if len(choice) > best_depth:
            best_depth, best_choice, best_part = len(choice), list(choice), part
    return best_depth, best_choice, best_part


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

def pseudo_dimension_detailed(
    M,
    *,
    points: str = "columns",
    exact: bool = True,
    time_limit: Optional[float] = None,
    max_thresholds: Optional[int] = None,
    restarts: int = 8,
    seed: int = 0,
) -> Result:
    """Compute the pseudo-dimension with full diagnostics.

    Parameters
    ----------
    M : array-like, shape (m, n)
        Real matrix, finite.  Rows are functions, columns are domain points.
    points : {"columns", "rows"}
        Which axis indexes the domain points; ``"rows"`` transposes first.
    exact : bool
        If False, return only the fast randomised greedy lower bound.
    time_limit : float, optional
        Seconds.  On expiry the best certified lower bound so far is returned
        with ``exact=False``.
    max_thresholds : int, optional
        Consider at most this many quantile-spaced thresholds per column.
        Makes wide continuous matrices tractable at the cost of exactness
        (the answer is still a valid lower bound, and ``exact`` is set False
        unless it happens to meet the upper bound).
    restarts : int
        Randomised greedy restarts used to seed the branch and bound.
    seed : int
        RNG seed, for reproducibility.
    """
    t_start = time.perf_counter()

    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {M.shape}")
    if points == "rows":
        M = M.T
    elif points != "columns":
        raise ValueError("points must be 'columns' or 'rows'")
    if np.issubdtype(M.dtype, np.floating) and M.size and not np.isfinite(M).all():
        raise ValueError("matrix contains NaN or inf")
    if max_thresholds is not None and max_thresholds < 1:
        raise ValueError("max_thresholds must be >= 1")

    def done(dim, is_exact, ub, wit, nodes):
        return Result(dim, is_exact, ub, wit, nodes, time.perf_counter() - t_start)

    m, n = M.shape
    if m == 0 or n == 0:
        return done(0, True, 0, Witness(), 0)

    # Duplicate rows can never separate two patterns, so collapse them; `rep`
    # maps each surviving row back to an original row index.
    Mu, rep = np.unique(M, axis=0, return_index=True)
    mu = Mu.shape[0]

    cols, values, R, truncated = _rank_encode(Mu, max_thresholds)
    ncols = len(cols)
    upper = min(ncols, mu.bit_length() - 1)             # d <= floor(log2 mu)
    if upper == 0:
        return done(0, True, upper, Witness(), 0)

    # Fixed search order: columns with more distinct splits first, so the
    # "depth + live columns remaining" cut-off bites early on the tail.
    order = sorted(range(ncols), key=lambda j: -values[j].size)
    cols = [cols[j] for j in order]
    values = [values[j] for j in order]
    R = np.ascontiguousarray(R[:, order])
    cols_R = [np.ascontiguousarray(R[:, j]) for j in range(ncols)]
    span = int(R.max()) + 1 if R.size else 1     # block-id stride for sorting

    def make_witness(choice: Sequence[Tuple[int, int]], part) -> Witness:
        # Present columns in original-matrix order; remap the pattern bits.
        pres = sorted(range(len(choice)), key=lambda k: cols[choice[k][0]])
        out_cols = [cols[choice[k][0]] for k in pres]
        out_thr = [float(values[choice[k][0]][choice[k][1]]) for k in pres]
        perm, offs = part[0], part[1]
        rows = []
        for p in range(offs.size):
            q = sum(((p >> pos) & 1) << k for pos, k in enumerate(pres))
            rows.append(int(rep[perm[offs[q]]]))
        return Witness(out_cols, out_thr, rows)

    # --- greedy seed --------------------------------------------------------
    rng = random.Random(seed)
    best, choice, part = _greedy_lower_bound(R, cols_R, mu, span, rng, restarts)
    best_witness = make_witness(choice, part) if best else Witness()

    if not exact or best >= upper:
        return done(best, best >= upper, upper, best_witness, 0)

    # --- exact branch and bound --------------------------------------------
    nodes = 0
    timed_out = False
    deadline = None if time_limit is None else t_start + time_limit
    stack: List[Tuple[int, int]] = []

    class _Timeout(Exception):
        pass

    class _Optimal(Exception):
        pass

    def dfs(depth: int, part, cand: np.ndarray) -> None:
        """``cand`` holds the candidate columns after the last chosen one that
        are still known to cut every block of every ancestor."""
        nonlocal best, best_witness, nodes
        nodes += 1
        if deadline is not None and (nodes & 0x3F) == 0 and time.perf_counter() > deadline:
            raise _Timeout

        if depth > best:
            best = depth
            best_witness = make_witness(stack, part)
            if best >= upper:
                raise _Optimal

        # A block of s rows survives at most floor(log2 s) further splits.
        room = int(part[2].min()).bit_length() - 1
        if depth + min(room, cand.size) <= best:
            return

        # Each child block must survive (best - depth) more splits, hence must
        # hold at least 2^(best - depth) rows.  The room test above guarantees
        # every block is at least twice that, so the order statistics exist.
        need = 1 << max(best - depth, 0)
        lo, hi, k1, k2 = _column_ranges(R, part, cand, need, span)
        alive = hi > lo                     # dead columns stay dead downstream
        cand = cand[alive]
        k1, k2 = k1[alive], k2[alive]

        n_live = cand.size
        if depth + min(n_live, room) <= best:
            return

        for idx in range(n_live):
            if depth + (n_live - idx) <= best:
                return                      # candidates only get scarcer
            jp = int(cand[idx])
            colv = cols_R[jp]
            rest = cand[idx + 1:]
            for k in range(int(k1[idx]), int(k2[idx]) + 1):
                # `best` may have grown since k1/k2 were computed, so re-check.
                nxt = _refine(colv, part, k, 1 << max(best - depth, 0))
                if nxt is None:
                    continue
                stack.append((jp, k))
                dfs(depth + 1, nxt, rest)
                stack.pop()

    try:
        dfs(0, _trivial_partition(mu), np.arange(ncols))
    except _Optimal:
        pass
    except _Timeout:
        timed_out = True

    is_exact = not timed_out and (not truncated or best >= upper)
    return done(best, is_exact, upper, best_witness, nodes)


def pseudo_dimension(M, **kwargs) -> int:
    """Pseudo-dimension of ``M``.  See :func:`pseudo_dimension_detailed`."""
    return pseudo_dimension_detailed(M, **kwargs).dimension


def vc_dimension(M, **kwargs) -> int:
    """VC dimension of a 0/1 matrix (rows = sets, columns = ground elements).

    Raises if ``M`` is not binary -- use :func:`pseudo_dimension` for
    real-valued matrices.
    """
    A = np.asarray(M)
    if A.size and not np.all((A == 0) | (A == 1)):
        raise ValueError("vc_dimension expects a 0/1 matrix; "
                         "use pseudo_dimension() for real-valued matrices")
    return pseudo_dimension(A, **kwargs)


# ---------------------------------------------------------------------------
# reference implementation (exponential; for testing only)
# ---------------------------------------------------------------------------

def _brute_force(M) -> int:
    """Definition-literal reference: try every column set and threshold tuple."""
    M = np.asarray(M)
    m, n = M.shape
    cand = [[float(v) for v in np.unique(M[:, j])[1:]] for j in range(n)]
    best = 0
    for d in range(1, min(n, max(m, 1).bit_length()) + 1):
        found = False
        for cs in combinations(range(n), d):
            if any(not cand[j] for j in cs):
                continue
            for thr in product(*(cand[j] for j in cs)):
                seen = set()
                for i in range(m):
                    seen.add(tuple(bool(M[i, j] >= t) for j, t in zip(cs, thr)))
                if len(seen) == 1 << d:
                    found = True
                    break
            if found:
                break
        if not found:
            break
        best = d
    return best


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    failures = 0

    def check(name, got, want):
        nonlocal failures
        ok = got == want
        failures += not ok
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}: got {got}, want {want}")

    print("fixed cases")

    # Full power set on d elements -> VC dim = d.
    for d in (1, 2, 3, 4, 5):
        P = np.array([[(i >> k) & 1 for k in range(d)] for i in range(1 << d)])
        check(f"power set d={d}", vc_dimension(P), d)

    # Singletons + empty set -> VC dim 1.
    S = np.vstack([np.zeros((1, 6), int), np.eye(6, dtype=int)])
    check("singletons on 6 points", vc_dimension(S), 1)

    # Half-lines (rows are step functions) -> VC dim 1.
    check("half-lines", vc_dimension(np.tril(np.ones((8, 8), int))), 1)

    # Intervals on 8 points -> VC dim 2.
    iv = [[1 if a <= x <= b else 0 for x in range(8)]
          for a in range(8) for b in range(a, 8)]
    check("intervals on 8 points", vc_dimension(np.array(iv)), 2)

    # Non-binary: a 2x2 value grid pseudo-shatters both columns.
    check("2x2 value grid",
          pseudo_dimension(np.array([[1., 2.], [2., 1.], [1., 1.], [2., 2.]])), 2)

    # Only 3 distinct rows -> capped at floor(log2 3) = 1.
    check("3x3 latin square",
          pseudo_dimension(np.array([[1., 2., 3.], [3., 1., 2.], [2., 3., 1.]])), 1)

    # Ordinal data whose third column needs a threshold away from the mean.
    O = np.array([[1., 1., 1.], [1., 1., 5.], [1., 9., 1.], [1., 9., 5.],
                  [7., 1., 1.], [7., 1., 5.], [7., 9., 1.], [7., 9., 5.]])
    check("3 independent ordinal columns", pseudo_dimension(O), 3)
    check("points='rows'", pseudo_dimension(O.T, points="rows"), 3)

    # A column needing a non-median threshold: the split must isolate value 2.
    T = np.array([[0., 0.], [0., 2.], [5., 0.], [5., 2.], [9., 1.], [1., 1.]])
    check("non-median threshold", pseudo_dimension(T), 2)

    # Degenerate shapes.
    check("constant matrix", pseudo_dimension(np.full((10, 5), 3.0)), 0)
    check("single row", pseudo_dimension(np.arange(9.).reshape(1, 9)), 0)
    check("single column", pseudo_dimension(np.arange(16.).reshape(16, 1)), 1)
    check("duplicate rows", pseudo_dimension(np.ones((64, 4))), 0)
    check("empty", pseudo_dimension(np.zeros((0, 3))), 0)

    print("randomised cross-check against brute force")
    rng = np.random.default_rng(12345)
    bad = 0
    for trial in range(120):
        m = int(rng.integers(2, 14))
        n = int(rng.integers(1, 6))
        k = int(rng.integers(2, 6))
        A = rng.integers(0, k, size=(m, n)).astype(float)
        fast, slow = pseudo_dimension(A), _brute_force(A)
        if fast != slow:
            bad += 1
            print(f"  [FAIL] trial {trial}: fast={fast} slow={slow}\n{A}")
    failures += bad
    print(f"  [{'ok  ' if bad == 0 else 'FAIL'}] 120 random matrices match brute force")

    print("witness validity")
    rng = np.random.default_rng(7)
    wbad = 0
    for _ in range(25):
        A = rng.integers(0, 4, size=(60, 8)).astype(float)
        res = pseudo_dimension_detailed(A)
        if res.dimension and not res.witness.verify(A):
            wbad += 1
    failures += wbad
    print(f"  [{'ok  ' if wbad == 0 else 'FAIL'}] 25 witnesses re-verified")

    print("truncated search is a valid lower bound")
    rng = np.random.default_rng(11)
    tbad = 0
    for _ in range(10):
        A = rng.normal(size=(80, 10))
        full = pseudo_dimension_detailed(A)
        cut = pseudo_dimension_detailed(A, max_thresholds=4)
        if cut.dimension > full.dimension or (cut.dimension and
                                              not cut.witness.verify(A)):
            tbad += 1
    failures += tbad
    print(f"  [{'ok  ' if tbad == 0 else 'FAIL'}] max_thresholds never overshoots")

    print("scale")
    rng = np.random.default_rng(3)
    for shape, lv in [((2000, 30), 2), ((5000, 40), 3), ((1000, 60), 8)]:
        A = rng.integers(0, lv, size=shape).astype(float)
        print(f"  {shape[0]}x{shape[1]}, {lv} levels -> "
              f"{pseudo_dimension_detailed(A, time_limit=20)}")
    A = rng.normal(size=(400, 25))
    print(f"  400x25 continuous, 8 cuts/col -> "
          f"{pseudo_dimension_detailed(A, max_thresholds=8, time_limit=20)}")

    print("FAILURES:" if failures else "all tests passed", failures or "")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="VC dimension / Pollard pseudo-dimension of a matrix.")
    ap.add_argument("path", nargs="?", help="CSV / delimited matrix file")
    ap.add_argument("--points", choices=("columns", "rows"), default="columns",
                    help="axis indexing the domain points (default: columns)")
    ap.add_argument("--delimiter", default=",")
    ap.add_argument("--skip-header", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--max-thresholds", type=int, default=None,
                    help="quantile-subsample thresholds per column (lower bound)")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--witness", action="store_true",
                    help="print the shattering certificate")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.path:
        ap.error("a matrix file is required (or use --selftest)")

    M = np.atleast_2d(np.genfromtxt(args.path, delimiter=args.delimiter,
                                    skip_header=args.skip_header, dtype=float))
    res = pseudo_dimension_detailed(M, points=args.points,
                                    time_limit=args.time_limit,
                                    max_thresholds=args.max_thresholds,
                                    restarts=args.restarts, seed=args.seed)
    print(f"matrix: {M.shape[0]} x {M.shape[1]}")
    print(res)
    if args.witness and res.witness and res.witness.dimension:
        w = res.witness
        print("columns:    ", w.columns)
        print("thresholds: ", w.thresholds, " (predicate: M[i, j] >= t)")
        print("rows:       ", w.rows)
        print("verified:   ", w.verify(M, points=args.points))
    return 0


if __name__ == "__main__":
    sys.exit(main())

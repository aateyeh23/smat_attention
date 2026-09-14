#!/usr/bin/env python3
"""Verify the claims of Section 3.1 that are checkable rather than timeable.

    python verify_theory.py --out-dir .            # everything (~15 min)
    python verify_theory.py --skip-exact-vc        # seconds

Writes four CSVs consumed by ``plot_smat.py``:

  theory_structure.csv   Theorem 3.1 (i),(ii),(iv) and the L_n-vs-I_n density
  theory_vc.csv          Theorem 3.1 (iii): exact VC, VC of the hyperplane
                         system C, and the constructive lower-bound witness
  theory_counting.csv    Table 1: I_prof, I_tok and the generic bound vs T
  theory_endpoint.csv    the d = floor(log2 T) endpoint and its admissibility

Deciding VC is LOGNP-complete, so the exact branch-and-bound only reaches small
T.  Three levels of evidence are recorded and kept distinct:

  witness      the construction's own shattered set, replayed against the
               definition -- a certified LOWER bound, available at every T;
  vc_C         the exact VC of the B x N0 hyperplane incidence, which the proof
               says is d-1 -- cheap because C is much smaller than M;
  vc_exact     the exact VC of the materialised T x T mask -- the real claim,
               affordable only for small T.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

from smat.mask import (build_mask, d_max_geometric, incidence_audit,
                       materialise, shatter_witness)
from smat.vc import pseudo_dimension_detailed


def write(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def hr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ---------------------------------------------------------------------------

def nnz_In_variant(spec, n_P: int) -> int:
    """What the previous ``I_n`` formulation would have had, at the same q, B.

    The top-left block contributed ``n`` nonzeros instead of ``n(n+1)/2``, which
    is why that formulation needed the global set ``P`` to reach Theta(T^2).
    """
    n, T_R = spec.n, spec.T_R
    lr = int(np.diff(spec.C_indptr)[np.arange(T_R) % spec.B].sum()
             * (spec.n_X / spec.N0))
    return n + T_R * (T_R + 1) // 2 + T_R * n_P + lr


def check_structure(args):
    hr("Theorem 3.1 (i),(ii),(iv): causal, unit diagonal, M <= L_T, nnz = Theta(T^2)")
    rows = []
    for T in args.struct_T:
        for d in range(1, min(6, d_max_geometric(T, chunk=64) + 1)):
            spec = build_mask(T, d, chunk=64)
            M = materialise(spec, np.int8)
            L = np.tril(np.ones((T, T), dtype=np.int8))
            nnz = int(M.sum())
            r = dict(
                T=T, d=d, n=spec.n, T_R=spec.T_R, q=spec.q, B=spec.B, N0=spec.N0,
                causal=bool(np.all(np.triu(M, 1) == 0)),
                unit_diagonal=bool(np.all(np.diag(M) == 1)),
                dominated_by_L_T=bool(np.all(M <= L)),
                is_L_T=bool(np.array_equal(M, L)),
                nnz=nnz, density=nnz / T ** 2,
                min_row_support=int(M.sum(1).min()),
                landmark_min_support=int(M[:spec.n].sum(1).min()),
                nnz_In_variant_P0=nnz_In_variant(spec, 0),
                nnz_In_variant_P8=nnz_In_variant(spec, 8),
            )
            r["density_In_variant_P0"] = r["nnz_In_variant_P0"] / T ** 2
            rows.append(r)
            print(f"  T={T:>6} d={d}: density {r['density']:.4f}  "
                  f"(L_n) vs {r['density_In_variant_P0']:.4f} (I_n, |P|=0)   "
                  f"min row support {r['min_row_support']} "
                  f"(landmark min {r['landmark_min_support']})  "
                  f"causal={r['causal']} <=L_T={r['dominated_by_L_T']}")
    write(os.path.join(args.out_dir, "theory_structure.csv"), rows)
    print("\n  Under I_n the top-left block gave n nonzeros, so density fell to")
    print("  ~1/8 and P was needed to carry it; under L_n it is ~1/4 from the")
    print("  two triangles alone, which is what pays for removing P.")
    return rows


def check_counting(args):
    hr("Table 1: I_prof = nnz(C) = Theta(T^{2-3/d}) vs I_tok = Theta(T^{2-2/d})")
    rows = []
    for d in args.count_d:
        for T in args.count_T:
            try:
                spec = build_mask(T, d, chunk=128, max_I_prof=args.max_I_prof)
            except (ValueError, NotImplementedError, MemoryError) as e:
                print(f"  T={T:>9,} d={d}: skipped ({e})")
                continue
            if spec.kind == "endpoint":
                continue
            a = incidence_audit(spec)
            rows.append(dict(
                T=T, d=d, q=spec.q, N0=spec.N0, B=spec.B, deg=spec.uniform_deg,
                I_prof=spec.I_prof, I_tok=spec.I_tok, pool_gain=a["pool_gain"],
                generic=T ** (2 - 1 / d), dense=T * T,
                build_ms=spec.build_times.get("total", 0.0) * 1e3,
                audit_exact=a["exact"], T_R_ge_2B=a.get("T_R_ge_2B", True)))
    write(os.path.join(args.out_dir, "theory_counting.csv"), rows)

    print(f"\n{'d':>3} {'quantity':>10} {'fitted alpha':>13} {'predicted':>10} "
          f"{'exact in q':>12}")
    for d in sorted({r["d"] for r in rows}):
        sub = [r for r in rows if r["d"] == d]
        if len(sub) < 3:
            continue
        Ts = np.log([r["T"] for r in sub])
        for key, pred in [("I_prof", 2 - 3 / d), ("I_tok", 2 - 2 / d)]:
            alpha = np.polyfit(Ts, np.log([r[key] for r in sub]), 1)[0]
            closed = ("q^(d-1)(q^(d-1)-1)/(q-1)" if key == "I_prof"
                      else "n_X (q^(d-1)-1)/(q-1)")
            print(f"{d:>3} {key:>10} {alpha:13.3f} {pred:10.3f} {closed:>28}")
    print("\n  q = nextprime(n^{1/d}) moves in coarse steps, so the counts are")
    print("  staircases in T and the fitted alpha approaches its limit from")
    print("  below; the closed forms above hold exactly at every T.")
    return rows


def check_vc(args):
    hr("Theorem 3.1 (iii): VC(M^(d)) = d")
    rows = []

    print("\n-- constructive lower bound (replayed against the definition) --")
    for d in args.count_d:
        for T in args.witness_T:
            try:
                spec = build_mask(T, d, chunk=128, max_I_prof=args.max_I_prof)
            except (ValueError, NotImplementedError, MemoryError):
                continue
            if spec.kind != "geometric":
                continue
            cols, ok, _ = shatter_witness(spec)
            rows.append(dict(T=T, d=d, kind=spec.kind, level="witness",
                             value=len(cols) if ok else -1, exact=ok,
                             seconds=0.0, ceiling=T.bit_length() - 1,
                             B=spec.B, N0=spec.N0, q=spec.q))
            print(f"  T={T:>7} d={d}: all 2^{len(cols)} patterns realised on "
                  f"{len(cols)} columns: {ok}")

    print("\n-- exact VC of the hyperplane incidence C  (the proof says d-1) --")
    for d in args.count_d:
        for T in args.vc_C_T:
            try:
                spec = build_mask(T, d, chunk=128, max_I_prof=args.max_I_prof)
            except (ValueError, NotImplementedError, MemoryError):
                continue
            if spec.kind != "geometric" or d == 1:
                continue
            if spec.B * spec.N0 > args.max_C_entries:
                continue          # check the size before materialising it
            C = spec.dense_C()
            t0 = time.time()
            res = pseudo_dimension_detailed(C, points="columns",
                                            time_limit=args.vc_time_limit)
            rows.append(dict(T=T, d=d, kind="C", level="vc_C",
                             value=res.dimension, exact=res.exact,
                             seconds=time.time() - t0, ceiling=d - 1,
                             B=spec.B, N0=spec.N0, q=spec.q))
            # a truncated search still returns a certified lower bound, so
            # "matches but not proven exact" is not the same as "wrong"
            flag = ("ok" if (res.dimension == d - 1 and res.exact)
                    else "lb" if res.dimension == d - 1 else "!!")
            print(f"  [{flag}] T={T:>7} d={d}: VC(C) = {res.dimension} "
                  f"(predicted {d - 1}, exact={res.exact}, "
                  f"{C.shape[0]}x{C.shape[1]}, {time.time() - t0:.1f}s)")

    if not args.skip_exact_vc:
        print("\n-- exact VC of the materialised T x T mask --")
        for T, d in args.exact_vc_cases:
            spec = build_mask(T, d, chunk=min(64, T // 8))
            M = materialise(spec, np.int8)
            t0 = time.time()
            res = pseudo_dimension_detailed(M, points="columns",
                                            time_limit=args.vc_time_limit)
            rows.append(dict(T=T, d=d, kind=spec.kind, level="vc_exact",
                             value=res.dimension, exact=res.exact,
                             seconds=time.time() - t0,
                             ceiling=T.bit_length() - 1,
                             B=spec.B, N0=spec.N0, q=spec.q))
            flag = ("ok" if (res.dimension == d and res.exact)
                    else ("lb" if res.dimension == d else "!!"))
            print(f"  [{flag}] T={T:>7} d={d}: VC = {res.dimension} "
                  f"(predicted {d}, exact={res.exact}, "
                  f"ceiling {T.bit_length() - 1}, {time.time() - t0:.1f}s)",
                  flush=True)

    write(os.path.join(args.out_dir, "theory_vc.csv"), rows)
    return rows


def check_endpoint(args):
    hr("the endpoint d = floor(log2 T): zeta landmarks, complemented Walsh rows")
    print("The draft places the shattered recent column at t = B+1 and so asks")
    print("for m >= 2B, which fails at an exact power of two (T_R = B).  The")
    print("*first* recent column works instead and needs only T_R >= B: every")
    print("recent row contains it and no zeta row does.  Both conditions are")
    print("recorded below, with the exact VC where the search is affordable.\n")
    rows = []
    for T in args.endpoint_T:
        d = T.bit_length() - 1
        spec = build_mask(T, d, chunk=1)
        cols, ok, _ = shatter_witness(spec)
        r = dict(T=T, d=d, k=spec.k, n=spec.n, T_R=spec.T_R, B=spec.B,
                 I_prof=spec.I_prof,
                 draft_condition_T_R_ge_2B=spec.T_R >= 2 * spec.B,
                 needed_condition_T_R_ge_B=spec.T_R >= spec.B,
                 witness_ok=ok, witness_cols=len(cols),
                 power_of_two=(T & (T - 1)) == 0)
        if T <= args.endpoint_vc_max_T:
            M = materialise(spec, np.int8)
            t0 = time.time()
            res = pseudo_dimension_detailed(M, points="columns",
                                            time_limit=args.vc_time_limit)
            r.update(vc=res.dimension, vc_exact=res.exact,
                     vc_seconds=time.time() - t0)
        rows.append(r)
        print(f"  T={T:>6} d={d:>2} k={spec.k:>2} n={spec.n:>5} T_R={spec.T_R:>5} "
              f"B={spec.B:>5}  T_R>=2B (draft): "
              f"{str(r['draft_condition_T_R_ge_2B']):>5}  T_R>=B (needed): "
              f"{str(r['needed_condition_T_R_ge_B']):>5}  witness: {str(ok):>5}" +
              (f"  exact VC = {r['vc']} (predicted {d})" if "vc" in r else ""))
    write(os.path.join(args.out_dir, "theory_endpoint.csv"), rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--skip-exact-vc", action="store_true")
    ap.add_argument("--vc-time-limit", type=float, default=300.0)
    ap.add_argument("--max-C-entries", type=int, default=4_000_000)
    ap.add_argument("--endpoint-vc-max-T", type=int, default=320)
    ap.add_argument("--max-I-prof", type=int, default=30_000_000,
                    help="skip (T, d) whose incidence exceeds this; the build "
                         "itself is Theta(nnz(C)) and d=5 at T=2^20 needs 4e8")
    args = ap.parse_args(argv)

    args.struct_T = [512, 1024, 2048, 4096]
    args.count_d = [1, 2, 3, 4, 5]
    args.count_T = [2 ** k for k in range(12, 21)]
    args.witness_T = [2 ** k for k in range(12, 21)]
    args.vc_C_T = [2 ** k for k in (12, 14, 16, 18, 20)]
    args.exact_vc_cases = [(128, 1), (128, 2), (256, 2), (512, 2),
                           (256, 3), (512, 3), (1024, 3),
                           (512, 4), (1024, 4), (2048, 4)]
    args.endpoint_T = [64, 100, 128, 200, 255, 256, 300, 511, 512, 600]

    os.makedirs(args.out_dir, exist_ok=True)
    check_structure(args)
    check_counting(args)
    check_endpoint(args)
    check_vc(args)
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())

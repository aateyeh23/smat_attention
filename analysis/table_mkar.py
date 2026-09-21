#!/usr/bin/env python3
"""Rebuild the multi-key subset recall table from the fair-protocol CSVs.

    python mkar_table.py            # the LaTeX table plus a plain summary
    python mkar_table.py --all      # every arm, including the diagnostic heads

One protocol throughout: batch 32, lr 3e-4, 12000 steps, T=1024, 8 pairs, one
layer, id hash.  The batch matters -- at batch 8 softmax never leaves its
plateau, which is what made every earlier baseline cell an artefact.

Rows are labelled by what the query is given.  "oracle" hands the query a
direction on which all k requested cells agree, so it measures what the MASK
admits; the learned heads measure whether a model can find that direction
unaided, and must be reported alongside it, never in place of it.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

RECURRENT = {"mamba2": "Mamba-2", "deltanet": "DeltaNet",
             "gated_deltanet": "Gated DeltaNet", "loglinear": "log-linear"}
ORDER = ["softmax", *RECURRENT.values(), "SMAT $d=1$",
         "SMAT $d=1$, memory matched",
         "LSH bucketing ($d=3$)", "LSH bucketing ($d=4$)",
         "SMAT $d=2$, oracle", "SMAT $d=3$, oracle", "SMAT $d=4$, oracle",
         # at d=2 the hyperplanes of F_q^1 are single points, so ContentAssign
         # collapses plane to point and the learned arm IS the oracle arm
         "SMAT $d=2$, learned",
         "SMAT $d=3$, single cell", "SMAT $d=4$, single cell",
         "SMAT $d=3$, learned", "SMAT $d=4$, learned",
         "SMAT $d=3$, learned (MLP)", "SMAT $d=4$, learned (MLP)",
         "SMAT $d=3$, learned (delta)", "SMAT $d=4$, learned (delta)",
         "SMAT $d=3$, learned (delta+soft)", "SMAT $d=4$, learned (delta+soft)",
         "SMAT $d=3$, learned (soft)"]
# The single-cell read is the LSH bucketing baseline: disjoint cells, VC 2.
MAIN = [r for r in ORDER[:11 + len(RECURRENT)] if r != "SMAT $d=2$, learned"]


def label(r):
    if r["arm"] == "softmax":
        return "softmax"
    if r["arm"] in RECURRENT:         # mask-free: its d only fixes the task geometry
        return RECURRENT[r["arm"]]
    if r["d"] == 1:
        # the memory-matched arm is d=1 with the feature map widened to r = N0*r
        return ("SMAT $d=1$, memory matched" if "_mem" in r["run"]
                else "SMAT $d=1$")
    if r["read_mode"] == "point":
        # a type is one cell, i.e. hashing into buckets and attending inside one
        return f"LSH bucketing ($d={r['d']}$)"
    if r["dirs"] == "oracle":
        return f"SMAT $d={r['d']}$, oracle"
    head = ("delta+soft" if "deltadir_soft" in r["run"] else
            "delta" if "deltadir" in r["run"] else
            "MLP" if "mlpdir" in r["run"] else
            "soft" if "_soft" in r["run"] else None)
    return (f"SMAT $d={r['d']}$, learned"
            + (f" ({head})" if head else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--all", action="store_true", help="include diagnostic heads")
    ap.add_argument("--steps", type=int, default=12000,
                    help="a run counts as finished at this step; shorter runs "
                         "are listed but not tabulated")
    ap.add_argument("--partial", action="store_true",
                    help="tabulate unfinished runs too, as the step they reached")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.results_dir, "mkar_fair2_*.csv")))
    if not files:
        raise SystemExit("no mkar_fair2_*.csv yet")
    frames = []
    for f in files:
        if os.path.getsize(f) == 0:          # a run that has not written yet
            continue
        frames.append(pd.read_csv(f, on_bad_lines="skip"))
    if not frames:
        raise SystemExit("no rows written yet")
    d = pd.concat(frames)
    d = d[d.step == d.groupby("run").step.transform("max")]
    # One run can be written by two campaigns (d=4 k=1 single-cell is in both
    # mkar_fair2_point.csv and mkar_fair2_point4.csv); counted twice, it would
    # weight that seed double in the mean.
    d = d.drop_duplicates(subset=["run", "step"])
    d["lab"] = d.apply(label, axis=1)
    # The CSVs carry no seed column; the run name does, and it is what makes two
    # rows repeats of each other rather than one row written twice.
    d["seed"] = d.run.str.extract(r"_s(\d+)_").astype(int)

    # A run that is still training has rows in the CSV from every evaluation so
    # far, and averaging one of those into a cell beside two finished seeds
    # reports a number that is neither.  With one seed per cell this only
    # understated a row; with three it silently mixes protocols.  Unfinished
    # runs are listed below the finished ones and left out of the table.
    short = d[d.step < a.steps]
    if not a.partial:
        d = d[d.step >= a.steps]
    if d.empty:
        raise SystemExit(f"no run has reached {a.steps} steps "
                         "(pass --partial to tabulate what there is)")

    piv = d.pivot_table(index="lab", columns="k", values="exact_support")
    reached = d.groupby("lab").step.min()
    seeds = d.groupby("lab").seed.apply(lambda s: sorted(set(s)))
    rows = [r for r in (ORDER if a.all else MAIN) if r in piv.index]
    missing = [r for r in piv.index if r not in ORDER]
    pending = (short.groupby(["lab", "seed"]).step.max()
               if not short.empty else None)

    # After filtering, every tabulated run is at --steps, so the step is only
    # worth printing when --partial put unfinished ones back in.
    print(f"% seeds behind each row{' and lowest step reached' if a.partial else ''}:")
    for r in rows:
        step = f"   step {int(reached[r]):5d}" if a.partial else ""
        print(f"%   {r:36s} n={len(seeds[r])} seeds {seeds[r]}{step}")
    if missing:
        print(f"% unlabelled rows present: {missing}")

    # A cell averaged over one seed and a cell averaged over three read the
    # same in the table, so the spread has to be printed somewhere.  Range, not
    # s.d.: at n = 3 the range is what was observed and an s.d. is a guess.
    rng = d.groupby(["lab", "k"]).exact_support.agg(["min", "max", "count"])
    wide = [r for r in rows if len(seeds[r]) > 1]
    if wide:
        print("% observed range over seeds (min-max), where n > 1:")
        for r in wide:
            cells = []
            for k in (1, 2, 3):
                if (r, k) in rng.index:
                    a = rng.loc[(r, k)]
                    cells.append(f"k{k} {a['min']:.3f}-{a['max']:.3f} (n{int(a['count'])})")
            print(f"%   {r:36s} " + "  ".join(cells))
    else:
        print("% every row is a single seed: the table has no spread to report.")
    if pending is not None and not a.partial:
        print(f"% still training, left out of the table (of {a.steps} steps):")
        for (lab, seed), st in pending.items():
            print(f"%   {lab:36s} seed {seed}  step {int(st)}")
    print()
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{l S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]}")
    print(r"\toprule")
    print(r"\textbf{Model} & {$k=1$} & {$k=2$} & {$k=3$} \\")
    print(r"\midrule")
    prev = None
    for r in rows:
        kind = r.split(",")[-1]
        if prev is not None and kind != prev:
            print(r"\midrule")
        prev = kind
        cells = []
        for k in (1, 2, 3):
            v = piv.loc[r].get(k)
            cells.append("{--}" if pd.isna(v) else f"{v:.3f}")
        print(f"{r} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    ns = sorted({len(seeds[r]) for r in rows})
    seed_note = ((f"Cells are means over {ns[0]} seeds. " if ns[0] > 1 else
                  "Every cell is a single seed. ") if len(ns) == 1 else
                 f"Cells are means over {ns[0]}--{ns[-1]} seeds; the count per "
                 r"row is listed in the comment above this table. ")
    print(r"\caption{Exact-support match at $T=1024$, $8$ pairs, one layer, "
          r"batch $32$, lr $3\times10^{-4}$, $12000$ steps. " + seed_note +
          r"\emph{oracle} hands "
          r"the query a direction on which all $k$ requested cells agree, and so "
          r"measures what the mask admits; \emph{learned} chooses it from the "
          r"query.}")
    print(r"\label{tab:mkar}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()

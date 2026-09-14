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

ORDER = ["softmax", "SMAT $d=1$",
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
MAIN = [r for r in ORDER[:8] if r != "SMAT $d=2$, learned"]


def label(r):
    if r["arm"] == "softmax":
        return "softmax"
    if r["d"] == 1:
        return "SMAT $d=1$"
    if r["read_mode"] == "point":
        return f"SMAT $d={r['d']}$, single cell"
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
    ap.add_argument("--steps", type=int, default=12000)
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
    d["lab"] = d.apply(label, axis=1)

    piv = d.pivot_table(index="lab", columns="k", values="exact_support")
    reached = d.groupby("lab").step.min()
    rows = [r for r in (ORDER if a.all else MAIN) if r in piv.index]
    missing = [r for r in piv.index if r not in ORDER]

    print("% steps reached per row (12000 = finished):")
    for r in rows:
        print(f"%   {r:36s} {int(reached[r])}")
    if missing:
        print(f"% unlabelled rows present: {missing}")
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
    print(r"\caption{Exact-support match at $T=1024$, $8$ pairs, one layer, "
          r"batch $32$, lr $3\times10^{-4}$, $12000$ steps. \emph{oracle} hands "
          r"the query a direction on which all $k$ requested cells agree, and so "
          r"measures what the mask admits; \emph{learned} chooses it from the "
          r"query. \emph{single cell} is the degenerate read in which a type is "
          r"one cell rather than a hyperplane.}")
    print(r"\label{tab:mkar}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()

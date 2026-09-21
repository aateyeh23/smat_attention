#!/usr/bin/env python3
"""The routing and multi-key recall tables as mean (std) over seeds.

    python analysis/table_meanstd.py [--results-dir results]

Both tables report exact support match in percent, one row per model and one
column per k.  The std is the sample standard deviation (ddof = 1) over seeds.
SMat rows are shaded.  Needs \\usepackage{float} and \\usepackage[table]{xcolor}.

Only finished runs count: a multi-key run must have reached --steps, and a run
written by two campaigns is counted once.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def cell(v):
    v = v.dropna()
    if v.empty:
        return "--"
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    return f"{v.mean():.2f} ({sd:.2f})" if len(v) > 1 else f"{v.mean():.2f}"


def emit(rows, ks, caption, label):
    out = [r"\begin{table}[H]", r"\centering",
           r"\begin{tabular}{l" + "r" * len(ks) + "}", r"\hline",
           "Model & " + " & ".join(f"$k={k}$" for k in ks) + r" \\", r"\hline"]
    for name, shade, vals in rows:
        if shade:
            out.append(r"\rowcolor{gray!12}")
        out.append(name + " & " + " & ".join(vals) + r" \\")
    out += [r"\hline", r"\end{tabular}", rf"\caption{{{caption}}}",
            rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(out)


def routing(rd):
    fs = ["routing_learned.csv", "routing_learned_recurrent.csv",
          "routing_learned_gdn_loglinear.csv"]
    d = pd.concat([pd.read_csv(os.path.join(rd, f)) for f in fs
                   if os.path.exists(os.path.join(rd, f))])
    d["pct"] = 100 * d.exact_support_frac
    ks = sorted(d.k.unique())
    spec = [("Softmax", "softmax", None, False),
            ("Softmax (content)", "softmax_content", None, False),
            ("Mamba-2", "mamba2", None, False),
            ("DeltaNet", "deltanet", None, False),
            ("Gated DeltaNet", "gated_deltanet", None, False),
            ("Log-Linear", "loglinear", None, False)]
    # d = 1 is the causal mask, so SMat d=1 is linear attention on the same
    # code path; it is a baseline, and is named and shaded as one.
    spec += [("Linear attention", "smat", 1, False)]
    spec += [(f"SMat ($d={D}$)", "smat", D, True)
             for D in sorted(d[d.arm == "smat"].d.unique()) if D > 1]
    rows, n = [], set()
    for name, arm, D, shade in spec:
        s = d[(d.arm == arm) & ((d.d == D) if D is not None else True)]
        if s.empty:
            continue
        n.update(s.groupby("k").seed.nunique())
        rows.append((name, shade, [cell(s[s.k == k].pct) for k in ks]))
    seeds = f"{min(n)}" if len(n) == 1 else f"{min(n)}--{max(n)}"
    cap = (r"Exact routing-pattern match (\%) on $d$-subset routing, by the "
           r"number $k$ of marked positions. $T=1024$, one layer, 1500 steps, "
           r"batch 64, learning rate 0.003; payloads are resampled each batch "
           r"and evaluation uses fresh ones. Linear attention is SMat at $d=1$, "
           r"where the mask is the causal triangle, run by the same code. "
           rf"Mean (std) over {seeds} seeds.")
    return emit(rows, ks, cap, "tab:routing")


def mkar(rd, steps):
    fr = [pd.read_csv(f, on_bad_lines="skip")
          for f in sorted(glob.glob(os.path.join(rd, "mkar_fair2_*.csv")))
          if os.path.getsize(f)]
    d = pd.concat(fr)
    d = d[d.step == d.groupby("run").step.transform("max")]
    d = d[d.step >= steps].drop_duplicates(subset=["run", "step"])
    d["pct"] = 100 * d.exact_support
    ks = [1, 2, 3]
    plane = (d.read_mode == "plane") & (d.dirs == "oracle")
    point = (d.read_mode == "point") & (d.dirs == "oracle")
    mem = d.run.str.contains("_mem")
    spec = [("Softmax", d.arm == "softmax", False),
            ("Mamba-2", d.arm == "mamba2", False),
            ("DeltaNet", d.arm == "deltanet", False),
            ("Gated DeltaNet", d.arm == "gated_deltanet", False),
            ("Log-Linear", d.arm == "loglinear", False),
            ("Linear attention", (d.arm == "smat") & (d.d == 1) & ~mem, False),
            ("Linear attention (memory matched)",
             (d.arm == "smat") & (d.d == 1) & mem, False)]
    spec += [(f"LSH bucketing ($d={D}$)",
              (d.arm == "smat") & (d.d == D) & point, False) for D in (3, 4)]
    spec += [(f"SMat ($d={D}$)", (d.arm == "smat") & (d.d == D) & plane
              & ~d.run.str.contains("dir") & ~mem, True) for D in (2, 3, 4)]
    rows, n = [], set()
    for name, m, shade in spec:
        s = d[m]
        if s.empty:                       # an arm with no finished run yet
            continue
        n.update(s.groupby("k").run.nunique())
        rows.append((name, shade, [cell(s[s.k == k].pct) for k in ks]))
    seeds = f"{min(n)}" if len(n) == 1 else f"{min(n)}--{max(n)}"
    cap = (r"Exact-support match (\%) on multi-key subset recall, by the "
           r"number $k$ of requested keys. $T=1024$, 8 pairs, one layer, batch "
           r"32, learning rate $3\times10^{-4}$, 12000 steps. SMat rows use the "
           r"oracle hyperplane direction, so they measure what the mask admits. "
           r"Every model shares one block (short convolution, projections, MLP) "
           r"and differs only in its sequence mixer; linear attention is SMat "
           r"at $d=1$. LSH bucketing is the degenerate read in which a type is "
           r"one cell rather than a hyperplane, so cells are disjoint. The "
           r"memory-matched row widens linear attention's feature map to "
           r"$r=N_0 r$, giving it the cache SMat holds at $d=2$ ($23\times$ the "
           r"default). "
           rf"Mean (std) over {seeds} seeds.")
    return emit(rows, ks, cap, "tab:mkar")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--steps", type=int, default=12000)
    a = ap.parse_args()
    print(routing(a.results_dir))
    print()
    print(mkar(a.results_dir, a.steps))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The paper's prefill figure (fig:prefill, figures/prefill_cost.png): prefill against T.

    python analysis/fig_prefill.py --results-dir results --outdir figures [--dark]

Reads the 4K-512K sweep results/results_bf16_512k_h200.csv, written by
experiments/jobs/run_prefill_512k.sbatch on one H200.  Bars are the interquartile
range of the repeats behind each point, drawn from the ``*_p25`` / ``*_p75``
columns that ``bench_prefill.py`` writes alongside each median.

Those columns are absent from any sweep run before they were added, and this
script says so and draws the medians alone rather than inventing a spread.  To
get the bars, rerun the sweep:

    experiments/submit.sh experiments/jobs/run_prefill_512k.sbatch results_bf16_512k_h200

An interquartile range, not a standard deviation: GPU timings have a long right
tail, and a symmetric bar around a median would misreport it.  The bars are
asymmetric for the same reason, and on a log axis a lower bar that reaches zero
would be unplottable anyway.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures import (MARKERS, apply_style, load, series, fit, dump, save,
                     set_T_axis, place_end_labels, theme)


def band(rows, d, key):
    """(T, median, lo, hi) for one series; lo/hi are None when unmeasured."""
    T, y = series(rows, d, f"{key}_ms")
    if not len(T):
        return T, y, None, None
    _, p25 = series(rows, d, f"{key}_ms_p25")
    _, p75 = series(rows, d, f"{key}_ms_p75")
    if len(p25) != len(T) or len(p75) != len(T):
        return T, y, None, None
    # errorbar() wants distances from the point, not absolute positions.
    return T, y, np.maximum(y - p25, 0.0), np.maximum(p75 - y, 0.0)


def draw(ax, rows, P):
    ds = sorted({r["d"] for r in rows})
    labels, out, measured = [], {}, False
    for i, d in enumerate(ds):
        T, y, lo, hi = band(rows, d, "prefill")
        if not len(T):
            continue
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        f = fit(T, y)
        lab = f"SMat $d$={d}" + (f"  ($\\alpha$={f[0]:.2f})" if f else "")
        # Timing spread is small (an interquartile range of 15 runs, usually
        # under 1% of the median), so the markers are kept small and the bars
        # heavy enough to show wherever the spread is resolvable at all.
        ax.errorbar(T, y, yerr=None if lo is None else [lo, hi],
                    color=c, marker=mk, ms=4, lw=1.6, capsize=4, capthick=1.4,
                    elinewidth=1.4, label=lab)
        labels.append((T[-1], y[-1], f"d={d}", c))
        out["T"], out[f"prefill_ms_d{d}"] = T, y
        if lo is not None:
            measured = True
            out[f"prefill_p25_d{d}"], out[f"prefill_p75_d{d}"] = y - lo, y + hi

    T, y, lo, hi = band(rows, ds[0], "sdpa")
    if len(T):
        f = fit(T, y)
        ax.errorbar(T, y, yerr=None if lo is None else [lo, hi],
                    color=P["muted"], marker="x", ms=4, lw=1.6, capsize=4,
                    capthick=1.4, elinewidth=1.4,
                    label="softmax" + (f"  ($\\alpha$={f[0]:.2f})" if f else ""))
        labels.append((T[-1], y[-1], "softmax", P["muted"]))
        out["softmax_ms"] = y
        if lo is not None:
            measured = True
            out["softmax_p25"], out["softmax_p75"] = y - lo, y + hi

    ax.set_xscale("log"); ax.set_yscale("log")
    set_T_axis(ax, sorted({r["T"] for r in rows}))
    ax.set_xlabel("context length $T$"); ax.set_ylabel("prefill (ms)")
    ax.set_title("Prefill cost, one layer")
    ax.legend(loc="upper left")
    place_end_labels(ax, labels)
    return out, measured


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--timing", default="results_bf16_512k_h200.csv")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--name", default="prefill_cost", help="output basename")
    a = ap.parse_args(argv)

    path = os.path.join(a.results_dir, a.timing)
    if not os.path.exists(path):
        print(f"no {path}; nothing to draw")
        return 1
    rows = load(path)
    P = theme(a.dark)
    if not a.dark:
        P["surface"] = "#ffffff"          # white, for the paper
    # The shared theme's palette, as in fig17: the d lines in colour and the
    # softmax baseline in the theme's muted grey.
    apply_style(P)
    os.makedirs(a.outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    out, measured = draw(ax, rows, P)
    if not measured:
        print("warning: no spread in %s: it predates the p25/p75 columns, so "
              "the points are medians with no error bars." % a.timing)
    save(fig, a.outdir, a.name)
    if out:
        keys = [k for k in out if k != "T"]
        dump(os.path.join(a.outdir, f"{a.name}.csv"),
             ["T"] + keys, [out["T"]] + [out[k] for k in keys])
    print(f"figures in {a.outdir}/  (error bars: {'yes' if measured else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

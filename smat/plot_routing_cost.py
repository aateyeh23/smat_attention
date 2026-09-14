#!/usr/bin/env python3
"""fig15: what the mask buys (left) against what it costs (right).

    python plot_routing_cost.py --results-dir results --outdir figs [--dark]

Left  exact routing-pattern match against the requested dimension k, from the
      learned runs -- routing_learned.csv and routing_learned_recurrent.csv,
      concatenated, which is sound because both were run with the same seed and
      flags and so share their marked column sets.
Right prefill against T for one attention layer, one line per d, from the timing
      sweep (results_bf16.csv), with full causal attention for reference.  This
      panel is a cost measurement on random inputs -- bench_smat.py trains
      nothing -- so its context lengths are unrelated to the task on the left.

Arms are read from the CSVs, so a rerun of train_routing.py with more arms or a
larger d needs no change here.  As written the left panel stops at d=4 because
d_max(1024)=4; the sweep has to move to T=16384 to reach d=6.

Every line is solid: dashes are reserved for nothing here, so series are told
apart by colour, marker and a direct label at the line end.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_smat import (LIGHT, DARK, MARKERS, apply_style, load, series, fit,
                       dump, save, set_T_axis, place_end_labels, theme)

# The baselines, in the order they should appear: (legend label, marker, shade,
# end label).  They carry one state and so a causal-prefix row support; keeping
# them in greys says that, and leaves the palette to mean d.
BASELINE = {
    "softmax":  ("softmax", "x", "muted", "softmax"),
    "mamba2":   ("Mamba-2", "v", "secondary", "Mamba-2"),
    "deltanet": ("DeltaNet", "P", "text", "DeltaNet"),
}


def panel_routing(ax, rows, P):
    """Exact routing-pattern match vs k, learned against the certified ceiling."""
    acc, ceil = defaultdict(dict), defaultdict(dict)
    for r in rows:
        arm, k = r["arm"], int(r["k"])
        key = (arm, int(r["d"])) if arm.startswith("smat") else (arm, None)
        acc[key].setdefault(k, []).append(100.0 * r["exact_support_frac"])
        if r.get("ceiling_answerable") is not None:
            ceil[key][k] = 100.0 * r["ceiling_answerable"]

    smat = sorted([k for k in acc if k[0] == "smat"], key=lambda t: t[1])
    labels, out = [], {}
    for i, key in enumerate(smat):
        d = key[1]
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        ks = sorted(acc[key])
        y = [float(np.mean(acc[key][k])) for k in ks]
        ax.plot(ks, y, color=c, marker=mk, label=f"SMAT $d$={d}")
        labels.append((ks[-1], y[-1], f"d={d}", c))
        out[f"smat_d{d}"] = y

    for arm, (lab, mk, shade, short) in BASELINE.items():
        key = (arm, None)
        if key not in acc:
            continue
        col = P[shade]
        ks = sorted(acc[key])
        y = [float(np.mean(acc[key][k])) for k in ks]
        ax.plot(ks, y, color=col, marker=mk, lw=1.6, label=lab)
        labels.append((ks[-1], y[-1], short, col))
        out[arm] = y

    ax.set_xlabel("requested routing dimension $k$")
    ax.set_ylabel("exact routing-pattern match (%)")
    ax.set_title("Exact routing-pattern match")
    ax.set_ylim(0, 104)
    ax.legend(loc="lower left", ncol=2)
    place_end_labels(ax, labels)
    return out


def panel_cost(ax, rows, P):
    ds = sorted({r["d"] for r in rows})
    labels, out = [], {}
    for i, d in enumerate(ds):
        T, y = series(rows, d, "prefill_ms")
        if not len(T):
            continue
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        f = fit(T, y)
        ax.plot(T, y, color=c, marker=mk,
                label=f"SMAT $d$={d}" + (f"  ($\\alpha$={f[0]:.2f})" if f else ""))
        labels.append((T[-1], y[-1], f"d={d}", c))
        out["T"], out[f"prefill_ms_d{d}"] = T, y
    T, y = series(rows, ds[0], "sdpa_ms")
    if len(T):
        f = fit(T, y)
        ax.plot(T, y, color=P["muted"], marker="x",
                label="softmax" + (f"  ($\\alpha$={f[0]:.2f})" if f else ""))
        labels.append((T[-1], y[-1], "softmax", P["muted"]))
        out["softmax_ms"] = y
    ax.set_xscale("log"); ax.set_yscale("log")
    set_T_axis(ax, sorted({r["T"] for r in rows}))
    ax.set_xlabel("context length $T$"); ax.set_ylabel("prefill (ms)")
    ax.set_title("Prefill cost, one layer")
    ax.legend(loc="upper left")
    place_end_labels(ax, labels)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--learned", nargs="+",
                    default=["routing_learned.csv",
                             "routing_learned_recurrent.csv"])
    ap.add_argument("--timing", default="results_bf16.csv")
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args(argv)

    P = dict(theme(a.dark))
    if not a.dark:
        P["surface"] = "#ffffff"          # white, for the paper
    apply_style(P)
    os.makedirs(a.outdir, exist_ok=True)

    learned = []
    for name in a.learned:
        path = os.path.join(a.results_dir, name)
        if os.path.exists(path):
            learned += load(path)
        else:
            print(f"  (skipped {name}: not present)")
    timing = load(os.path.join(a.results_dir, a.timing))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    left = panel_routing(axL, learned, P)
    right = panel_cost(axR, timing, P)
    fig.tight_layout()
    save(fig, a.outdir, "fig15_routing_cost")

    ks = sorted({int(r["k"]) for r in learned})
    dump(os.path.join(a.outdir, "fig15_routing_cost_left.csv"),
         ["k"] + list(left), [ks] + [left[c] for c in left])
    dump(os.path.join(a.outdir, "fig15_routing_cost_right.csv"),
         list(right), [right[c] for c in right])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""fig17: exact routing-pattern match, the left panel of fig15 on its own.

    python analysis/fig_routing.py --results-dir results --outdir figures [--dark]

Differences from fig15's left panel, all deliberate:

  * Every VC-1 baseline is drawn except content-addressed softmax: softmax,
    Mamba-2, DeltaNet, Gated DeltaNet and log-linear attention.  DeltaNet was
    dropped from an earlier version and is back; it tracks log-linear and SMAT
    d=1 closely, which is the theory's own prediction -- M^(1) is the causal
    mask, so d=1 is linear attention and all of them have VC dimension 1.
  * Only the SMAT lines are labelled, at their ends and in a larger size, and
    they have no legend entry.  The baselines are in greys, named in the legend.
  * Colours are the shared theme's palette, deliberately lighter than fig16's.
  * Gated DeltaNet and log-linear attention are added.  Prop. vc1 predicts both
    at VC 1, and until now that was the only support for it: DeltaNet and
    Mamba-2 were measured and these two were argued.  Log-linear attention is
    the sharper of the pair, because it is the one model here whose state count
    grows with T -- Theta(log T) of them -- so if carrying more were what lifts
    the ceiling, it is the line that would show it.
  * Error bars, which the panel has the data for and did not draw: every point
    is three seeds, and fig15 plotted their mean with no indication of spread.

Bars span the three seeds (min to max), not a standard deviation: with n=3 an
s.d. is a poor estimate and implies a distribution nobody checked, while the
range is exactly what was observed.  They are asymmetric for the same reason.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures import (MARKERS, apply_style, load, dump, save, place_end_labels,
                     theme)

# Baselines in legend order: (legend label, marker, grey index, dash).  Their
# row support is the causal prefix, so their VC dimension is 1 whatever they
# carry -- one state, one gated state, or log T of them.  All of them are grey,
# which says that and leaves colour to mean d; within the greys they are told
# apart by marker and dash.  Markers avoid MARKERS, which the d lines use.
BASELINE = {
    "softmax":         ("Softmax", "x", 2, "-"),
    "mamba2":          ("Mamba-2", "h", 1, "-"),
    "deltanet":        ("DeltaNet", "p", 1, ":"),
    "gated_deltanet":  ("Gated DeltaNet", "P", 0, "--"),
    "loglinear":       ("Log-Linear", "*", 0, "-."),
}


def spread(vals):
    """(mean, lower distance, upper distance) over the seeds at one point."""
    a = np.asarray(vals, dtype=float)
    m = float(a.mean())
    return m, m - float(a.min()), float(a.max()) - m


def draw(ax, rows, P, greys):
    acc = defaultdict(dict)
    for r in rows:
        arm = r["arm"]
        key = (arm, int(r["d"])) if arm.startswith("smat") else (arm, None)
        acc[key].setdefault(int(r["k"]), []).append(100.0 * r["exact_support_frac"])

    labels, out, n_seeds = [], {}, set()

    # SMAT, one line per d, in the palette.  These keep their end labels.
    smat = sorted([k for k in acc if k[0] == "smat"], key=lambda t: t[1])
    for i, key in enumerate(smat):
        d = key[1]
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        ks = sorted(acc[key])
        stats = [spread(acc[key][k]) for k in ks]
        y = [s[0] for s in stats]
        lo = [s[1] for s in stats]
        hi = [s[2] for s in stats]
        n_seeds.update(len(acc[key][k]) for k in ks)
        # no legend entry: the end label names the line
        ax.errorbar(ks, y, yerr=[lo, hi], color=c, marker=mk, lw=1.8,
                    capsize=3, elinewidth=1.1, zorder=3, label="_nolegend_")
        labels.append((ks[-1], y[-1], f"d={d}", c))
        out[f"smat_d{d}"] = y
        out[f"smat_d{d}_min"] = [a - b for a, b in zip(y, lo)]
        out[f"smat_d{d}_max"] = [a + b for a, b in zip(y, hi)]

    # Baselines in greys, legend only -- no end label.
    for arm, (lab, mk, g, ls) in BASELINE.items():
        key = (arm, None)
        if key not in acc:
            continue
        col = greys[g]
        ks = sorted(acc[key])
        stats = [spread(acc[key][k]) for k in ks]
        y = [s[0] for s in stats]
        n_seeds.update(len(acc[key][k]) for k in ks)
        ax.errorbar(ks, y, yerr=[[s[1] for s in stats], [s[2] for s in stats]],
                    color=col, marker=mk, ls=ls, lw=1.3, ms=6, capsize=2.5,
                    elinewidth=0.9, zorder=2, label=lab)
        out[arm] = y
        out[f"{arm}_min"] = [s[0] - s[1] for s in stats]
        out[f"{arm}_max"] = [s[0] + s[2] for s in stats]

    ax.set_xlabel("requested routing dimension $k$")
    ax.set_ylabel("exact routing-pattern match (%)")
    ax.set_title("Exact routing-pattern match")
    ax.set_ylim(0, 104)
    ax.legend(loc="lower left", ncol=2, handlelength=2.6)
    # larger than the shared default: these are the only direct labels
    place_end_labels(ax, labels, min_gap_pt=15.0, fontsize=12)
    return out, sorted(n_seeds)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--learned", nargs="+",
                    default=["routing_learned.csv",
                             "routing_learned_recurrent.csv",
                             "routing_learned_gdn_loglinear.csv"])
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args(argv)

    P = theme(a.dark)
    if not a.dark:
        P["surface"] = "#ffffff"          # white, for the paper
    # the shared theme's own palette and greys, not fig16's darker one
    greys = [P["text"], P["secondary"], P["muted"]]
    apply_style(P)
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for name in a.learned:
        path = os.path.join(a.results_dir, name)
        if os.path.exists(path):
            rows += load(path)
        else:
            print(f"  (skipped {name}: not present)")
    if not rows:
        print("no learned-routing CSVs; nothing to draw")
        return 1

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    out, n_seeds = draw(ax, rows, P, greys)
    fig.tight_layout()
    save(fig, a.outdir, "fig17_routing")

    ks = sorted({int(r["k"]) for r in rows})
    if out:
        dump(os.path.join(a.outdir, "fig17_routing.csv"),
             ["k"] + list(out), [ks] + [out[c] for c in out])
    if n_seeds != [3]:
        print(f"warning: seeds per point vary or are not 3: {n_seeds}")
    print(f"figures in {a.outdir}/  (seeds per point: {n_seeds})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

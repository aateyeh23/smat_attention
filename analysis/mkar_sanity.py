#!/usr/bin/env python3
"""Sanity checks behind the multi-key subset recall table, from CSVs already on disk.

    python analysis/mkar_sanity.py                 # LaTeX tables on stdout, figure to figures/

Four things a reader needs before trusting a table whose baselines are all 0
and whose softmax row has a 50-point spread:

  per-seed      every seed of every arm, so the spread is visible as outcomes
                rather than summarised as a standard deviation
  curves        training loss and exact-support per seed against step, with the
                loss of the constant predictor that outputs the base rate k/N
                for every pair, i.e. the plateau a model sits on until it
                learns to use the query  (fig18_mkar_sanity)
  prior gap     how far each baseline's final loss is from that constant
  posctrl       the same harness, metric and budget at T = 128 and 256, where
                the baselines are expected to solve k = 1
  softmax sweep the learning-rate / warmup settings tried for softmax

The loss is binary cross-entropy with logits over the N = 8 pair outputs, so
the constant predictor's loss is the binary entropy H(k/N) nats; exact support
thresholds the raw logit at 0.5 and asks the thresholded vector to equal the
k-hot target.  A constant output is all-zero or all-one after any threshold,
never k-hot for 1 <= k < N, so it scores exactly 0.
"""
from __future__ import annotations

import glob
import math
import os
import sys

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures import LIGHT                                    # noqa: E402

N_PAIRS = 8
STEPS = 12000
KS = (1, 2, 3)
LINEAR = ["Mamba-2", "DeltaNet", "Gated DeltaNet", "Log-Linear",
          "Linear attention", "Linear attention (memory matched)"]
ROWS = ["Softmax", *LINEAR, "LSH bucketing (121 buckets)", "LSH bucketing (125 buckets)",
        "SMat ($d=2$)", "SMat ($d=3$)", "SMat ($d=4$)"]
REC = {"mamba2": "Mamba-2", "deltanet": "DeltaNet",
       "gated_deltanet": "Gated DeltaNet", "loglinear": "Log-Linear"}


def prior_loss(k, n=N_PAIRS):
    """BCE of the constant predictor sigmoid(logit) = k/n, i.e. H(k/n) in nats."""
    p = k / n
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def label(r):
    """The paper's row names; SMat rows are the oracle-direction arms."""
    if r.arm == "softmax":
        return "Softmax"
    if r.arm in REC:
        return REC[r.arm]
    if r.d == 1:
        return "Linear attention (memory matched)" if "_mem" in r.run else "Linear attention"
    if r.read_mode == "point":
        # d=3 hashes into q^2 = 121 cells, d=4 into q^3 = 125
        return f"LSH bucketing ({ {3: 121, 4: 125}[r.d] } buckets)"
    if r.dirs == "oracle":
        return f"SMat ($d={r.d}$)"
    return None


def load(pattern):
    fs = [f for f in sorted(glob.glob(pattern)) if os.path.getsize(f)]
    d = pd.concat([pd.read_csv(f, on_bad_lines="skip") for f in fs])
    d = d.drop_duplicates(subset=["run", "step"])
    d["seed"] = d.run.str.extract(r"_s(\d+)_").astype(int)
    d["T"] = d.run.str.extract(r"_T(\d+)_").astype(int)
    return d


def fmt(v):
    return f"{100 * v:.1f}"


def per_seed_table(d):
    fin = d[d.step == STEPS].copy()
    fin["lab"] = fin.apply(label, axis=1)
    fin = fin[fin.lab.notna()]
    print(r"\begin{table}[h]")
    print(r"\centering\scriptsize\setlength{\tabcolsep}{3pt}")
    print(r"\begin{tabular}{l" + "rrr" * len(KS) + "}")
    print(r"\toprule")
    print(" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$k={k}$}}" for k in KS) + r" \\")
    print(" ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(KS))))
    print("Model & " + " & ".join(["s0 & s1 & s2"] * len(KS)) + r" \\")
    print(r"\midrule")
    for lab in ROWS:
        cells = []
        for k in KS:
            for s in (0, 1, 2):
                v = fin[(fin.lab == lab) & (fin.k == k) & (fin.seed == s)].exact_support
                cells.append("{--}" if v.empty else fmt(v.iloc[0]))
        print(f"{lab} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Per-seed exact-support accuracy (\%) behind Table~\ref{tab:mkar}, "
          r"after 12K steps on 2{,}048 fresh queries per run. SMat rows use rule-chosen "
          r"hyperplane directions (see text).}")
    print(r"\label{tab:mkar-seeds}")
    print(r"\end{table}")
    print()

    lin = fin[fin.lab.isin(LINEAR)].copy()
    lin["gap"] = (lin.loss - lin.k.map(prior_loss)).abs()
    print(f"% {len(lin)} linear-baseline runs; max |final loss - H(k/8)| = {lin.gap.max():.4f} nats")
    for k in KS:
        print(f"%   k={k}: H(k/8) = {prior_loss(k):.4f}; final losses "
              f"{lin[lin.k == k].loss.min():.4f}-{lin[lin.k == k].loss.max():.4f}; "
              f"base-rate logit {math.log(k / (N_PAIRS - k)):+.3f}")
    print()


def curves(d, out="figures/fig18_mkar_sanity"):
    d = d.copy()
    d["lab"] = d.apply(label, axis=1)
    c = LIGHT
    arms = [("Softmax", c["series"][0], lambda x: x.lab == "Softmax"),
            ("Linear baselines (6 models)", c["series"][1], lambda x: x.lab.isin(LINEAR)),
            ("SMat $d=4$ (rule-chosen directions)", c["series"][2], lambda x: x.lab == "SMat ($d=4$)")]
    markers = {0: "o", 1: "s", 2: "^"}
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": c["muted"],
                         "axes.labelcolor": c["secondary"], "xtick.color": c["secondary"],
                         "ytick.color": c["secondary"]})
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharex=True)
    rows = []
    for j, k in enumerate(KS):
        la, ea = axes[0, j], axes[1, j]
        for name, col, sel in arms:
            sub = d[sel(d) & (d.k == k)]
            for run, g in sub.groupby("run"):
                g = g.sort_values("step")
                s = int(g.seed.iloc[0])
                kw = dict(color=col, lw=1.2, marker=markers[s], ms=3.5,
                          alpha=0.55 if name.startswith("Linear") else 0.95)
                la.plot(g.step / 1000, g.loss, **kw)
                ea.plot(g.step / 1000, 100 * g.exact_support, **kw)
                rows += [dict(arm=name, run=run, k=k, seed=s, step=int(r.step),
                              loss=r.loss, exact_support=r.exact_support)
                         for r in g.itertuples()]
        la.axhline(prior_loss(k), color=c["muted"], ls="--", lw=1.0)
        la.text(2.2, prior_loss(k) * 0.45, f"constant prior $H({k}/8)$",
                color=c["secondary"], fontsize=7, va="top")
        la.set_yscale("log"); la.set_ylim(1e-6, 2.0)
        ea.set_ylim(-5, 105)
        la.set_title(f"$k={k}$", color=c["text"], fontsize=9)
        for ax in (la, ea):
            ax.grid(True, color=c["grid"], lw=0.6); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        ea.set_xlabel("training step (K)")
    axes[0, 0].set_ylabel("train BCE (nats)")
    axes[1, 0].set_ylabel("exact support (%)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=col, lw=1.4) for _, col, _ in arms]
    handles += [Line2D([], [], color=c["secondary"], marker=m, ls="", ms=4)
                for m in markers.values()]
    labels = [a[0] for a in arms] + [f"seed {s}" for s in markers]
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(out + ".csv", index=False)
    print(f"% wrote {out}.pdf/.png/.csv")
    print()


def posctrl_table(d, ref):
    d = d[d.step == STEPS].copy()
    d["lab"] = d.apply(label, axis=1)
    ref = ref[(ref.step == STEPS) & (ref.k == 1) & (ref.seed == 0)].copy()
    ref["lab"] = ref.apply(label, axis=1)
    arms = ["Softmax", "Mamba-2", "DeltaNet", "Gated DeltaNet", "Log-Linear", "Linear attention"]
    print(r"\begin{table}[h]")
    print(r"\centering\small")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Model & $T=128$ & $T=256$ & $T=1024$ \\")
    print(r"\midrule")
    for lab in arms:
        cells = []
        for T, src in ((128, d), (256, d), (1024, ref)):
            v = src[(src.lab == lab) & (src["T"] == T)].exact_support
            cells.append("{--}" if v.empty else fmt(v.iloc[0]))
        print(f"{lab} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Positive control: exact-support accuracy (\%) for $k=1$, seed 0, "
          r"with the harness, metric, model and 12K-step budget of Table~\ref{tab:mkar} and "
          r"only the context length changed (8 pairs throughout).}")
    print(r"\label{tab:mkar-posctrl}")
    print(r"\end{table}")
    print()


def softmax_sweep(base):
    cfgs = [("no warmup, $3\\times10^{-4}$ (reported)", base[base.arm == "softmax"])]
    for lr, tex in (("1e-4", "10^{-4}"), ("3e-4", "3\\times10^{-4}"), ("1e-3", "10^{-3}")):
        cfgs.append((f"1K-step warmup, ${tex}$", load(f"results/mkar_softsweep_wu_lr{lr}.csv")))
    print(r"\begin{table}[h]")
    print(r"\centering\small")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Softmax schedule, LR & $k=1$ & $k=2$ & $k=3$ \\")
    print(r"\midrule")
    for name, g in cfgs:
        g = g[g.step == STEPS]
        cells = []
        for k in KS:
            v = g[g.k == k].sort_values("seed").exact_support
            cells.append(" / ".join(fmt(x) for x in v))
        print(f"{name} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Softmax on multi-key subset recall at $T=1024$: exact-support accuracy (\%) "
          r"for seeds 0 / 1 / 2 under each schedule tried. A run that leaves the "
          r"constant-prior plateau reaches $\geq 99\%$; the runs between 1\% and 25\% had "
          r"begun leaving it when training stopped. The reported row is the best schedule.}")
    print(r"\label{tab:mkar-softmax-sweep}")
    print(r"\end{table}")


def main():
    main_ = load("results/mkar_fair2_*.csv")
    main_ = main_[main_["T"] == 1024]
    per_seed_table(main_)
    curves(main_)
    posctrl_table(load("results/mkar_posctrl.csv"), main_)
    softmax_sweep(main_)


if __name__ == "__main__":
    main()

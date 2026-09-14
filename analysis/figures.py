#!/usr/bin/env python3
"""Every figure in the paper, from the bench, verify and routing CSVs.

    python plot_smat.py results/results_bf16.csv --results-dir results
    python plot_smat.py results/results_bf16.csv --results-dir results --dark

With `--results-dir` pointing at the directory the jobs wrote, one invocation
regenerates all thirteen panels; each is skipped, with a line saying so, when
its CSV is absent.  Produces PDF (for the paper) and PNG (for a quick look):

  from the timing sweep (bench_smat.py)
    fig1_scaling      prefill vs T against causal SDPA, with fitted exponents
    fig2_incidence    the incidence phase alone: pooled 2-3/d vs direct 2-2/d
    fig3_phases       where prefill time goes, one panel per d
    fig4_decode       per-token cost (flat in T) and cache size vs a KV cache
    fig8_triton       the fused kernels against the torch path, time and memory
    fig9_subtractive  Remark 3.10: retained mass and the error it costs
    fig10_rank        sensitivity to the feature rank r  (results_r*.csv)
    fig11_chunk       sensitivity to the chunk width c   (results_c*.csv)

  from the theory sweep (verify_theory.py)
    fig5_counting     Table 1: I_prof, I_tok, the generic bound, and T^2
    fig6_vc           Theorem 3.1(iii): measured VC against d
    fig7_density      Theorem 3.1(iv): L_n vs the previous I_n block

  from the expressivity benchmark (bench_routing.py, train_routing.py)
    fig12_routing         Pi_M(k), the answerable fraction, task error vs k
    fig13_routing_learned the learned run against that certified ceiling

Every panel also writes a companion .csv of the exact plotted values, so the
numbers behind a figure can be read without re-running anything.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker            # set_T_axis reaches for NullFormatter

# Categorical slots 1-5 of the reference palette, validated for both modes on
# the adjacent pairlist (lines and stacks).  Light mode WARNs on contrast for
# aqua/yellow/magenta, so every series also carries a direct label and its own
# marker -- the relief rule, and secondary encoding for CVD.
LIGHT = dict(
    series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    surface="#fcfcfb", text="#0b0b0b", secondary="#52514e", muted="#8a8880",
    grid="#e6e5e1",
)
DARK = dict(
    series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
    surface="#1a1a19", text="#ffffff", secondary="#c3c2b7", muted="#8a8880",
    grid="#33322f",
)
MARKERS = ["o", "s", "^", "D", "v"]

# The learned run's four arms.  These are compared against each other rather
# than against d, so they are separated by dash pattern and marker, not colour.
ARM_STYLE = {
    "smat": ("-", "o", 1.0),
    "smat_content": ("-.", "s", 0.8),
    "softmax": ("--", "x", 0.75),
    "softmax_content": (":", "^", 0.9),
}


def theme(dark: bool) -> Dict:
    return DARK if dark else LIGHT


def apply_style(P: Dict):
    plt.rcParams.update({
        "figure.facecolor": P["surface"], "axes.facecolor": P["surface"],
        "savefig.facecolor": P["surface"],
        "text.color": P["text"], "axes.labelcolor": P["secondary"],
        "xtick.color": P["secondary"], "ytick.color": P["secondary"],
        "axes.edgecolor": P["grid"], "grid.color": P["grid"],
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8.5,
        "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "figure.dpi": 140,
    })


def pow2(x, _=None):
    """Tick labels as 4k / 64k / 256k rather than 10^5.3."""
    if x >= 1024 and float(x).is_integer():
        k = x / 1024
        return f"{int(k)}k" if k == int(k) else f"{k:g}k"
    return f"{x:g}"


# ---------------------------------------------------------------------------

def load(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if v == "" or v is None:
                    out[k] = None
                    continue
                try:
                    out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                except ValueError:
                    out[k] = v
            rows.append(out)
    return rows


def series(rows, d, key):
    pts = sorted((r["T"], r[key]) for r in rows
                 if r["d"] == d and r.get(key) is not None)
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def fit(T, y):
    if len(T) < 3:
        return None
    a, b = np.polyfit(np.log(T), np.log(y), 1)
    return float(a), float(b)


def dump(path, header, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(zip(*cols))


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def set_T_axis(ax, Ts):
    """Tick only at the measured context lengths, labelled 4k / 64k / 256k.

    A log axis otherwise decorates itself with 2x10^3-style minor labels, which
    say nothing here -- the x values are powers of two by construction.
    """
    Ts = sorted({int(t) for t in Ts})
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([pow2(t) for t in Ts])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xmargin(0.10)


def place_end_labels(ax, items, min_gap_pt=11.0):
    """Direct labels at line ends, pushed apart so converging lines stay legible.

    The relief rule requires visible labels for the low-contrast slots, so these
    are not optional decoration -- they have to actually be readable.
    """
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    px_per_pt = fig.dpi / 72.0
    disp = [ax.transData.transform((x, y)) for x, y, _, _ in items]
    order = sorted(range(len(items)), key=lambda i: disp[i][1])
    ys = [disp[i][1] for i in order]
    gap = min_gap_pt * px_per_pt
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    for k, i in enumerate(order):
        x, y, text, color = items[i]
        ax.annotate(text, xy=(x, y),
                    xytext=(5, (ys[k] - disp[i][1]) / px_per_pt),
                    textcoords="offset points", color=color, fontsize=8.5,
                    va="center", ha="left", fontweight="bold",
                    annotation_clip=False)


# ---------------------------------------------------------------------------
# timing figures
# ---------------------------------------------------------------------------

def fig_scaling(rows, P, outdir):
    ds = sorted({r["d"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    labels, header, cols = [], ["T"], []
    allT = sorted({r["T"] for r in rows})
    header_done = False
    for i, d in enumerate(ds):
        T, y = series(rows, d, "prefill_ms")
        if not len(T):
            continue
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        f = fit(T, y)
        lab = f"SMAT d={d}" + (f"  ($\\alpha$={f[0]:.2f})" if f else "")
        ax.plot(T, y, color=c, marker=mk, label=lab)
        labels.append((T[-1], y[-1], f"d={d}", c))
        if not header_done:
            cols.append(T); header_done = True
        header.append(f"prefill_ms_d{d}")
        cols.append(y)
    T, y = series(rows, ds[0], "sdpa_ms")
    if len(T):
        f = fit(T, y)
        ax.plot(T, y, color=P["muted"], marker="x", ls="--",
                label="causal SDPA" + (f"  ($\\alpha$={f[0]:.2f})" if f else ""))
        labels.append((T[-1], y[-1], "SDPA", P["muted"]))
    ax.set_xscale("log"); ax.set_yscale("log")
    set_T_axis(ax, allT)
    ax.set_xlabel("context length $T$"); ax.set_ylabel("prefill (ms)")
    ax.set_title("Prefill cost against causal attention")
    ax.legend(loc="upper left")
    place_end_labels(ax, labels)
    fig.text(0.0, -0.06,
             "Total prefill is a $\\Theta(T)$ scan plus a $\\Theta(T^{2-3/d})$ "
             "long-range term, so $\\alpha\\!\\approx\\!1$ for $d \\leq 3$; "
             "fig2 isolates the long-range term.",
             fontsize=8, color=P["secondary"])
    save(fig, outdir, "fig1_scaling")
    if cols:
        dump(os.path.join(outdir, "fig1_scaling.csv"), header, cols)


def fig_incidence(rows, P, outdir):
    """The claim of Sec. 3.3: pooling buys a factor T^{1/d} over a direct scatter."""
    ds = [d for d in sorted({r["d"] for r in rows}) if d >= 2]
    if not ds:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    allT = sorted({r["T"] for r in rows})
    for i, d in enumerate(ds):
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        Tp, yp = series(rows, d, "phase_lr_incidence_ms")
        Td, yd = series(rows, d, "phase_direct_incidence_ms")
        if len(Tp):
            f = fit(Tp, yp)
            axes[0].plot(Tp, yp, color=c, marker=mk,
                         label=f"d={d} pooled ($\\alpha$={f[0]:.2f}, "
                               f"pred {2 - 3/d:.2f})" if f else f"d={d} pooled")
        if len(Td):
            f = fit(Td, yd)
            axes[0].plot(Td, yd, color=c, marker=mk, ls="--", alpha=0.75,
                         label=f"d={d} direct ($\\alpha$={f[0]:.2f}, "
                               f"pred {2 - 2/d:.2f})" if f else f"d={d} direct")
        Ti, yi = series(rows, d, "I_prof")
        Tt, yt = series(rows, d, "I_tok")
        if len(Ti):
            axes[1].plot(Ti, yi, color=c, marker=mk, label=f"d={d}  $I_{{prof}}$")
            axes[1].plot(Tt, yt, color=c, marker=mk, ls="--", alpha=0.75,
                         label=f"d={d}  $I_{{tok}}$")
    for ax, ttl, yl in ((axes[0], "measured incidence phase", "ms"),
                        (axes[1], "the counts themselves", "incidences")):
        ax.set_xscale("log"); ax.set_yscale("log")
        set_T_axis(ax, allT)
        ax.set_xlabel("context length $T$"); ax.set_ylabel(yl)
        ax.set_title(ttl); ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Applying $C$: pooled $\\Theta(T^{2-3/d})$ vs direct "
                 "$\\Theta(T^{2-2/d})$", y=1.01)
    fig.text(0.0, -0.06,
             "Left is wall clock, right is the exact incidence count from the "
             "built structure.  A phase pinned at the kernel-launch floor fits "
             "$\\alpha\\!\\approx\\!0$ however small the count is; the right "
             "panel is free of that.", fontsize=8, color=P["secondary"])
    save(fig, outdir, "fig2_incidence")


PHASES = [("phase_scan_ms", "causal scan"),
          ("phase_lr_pool_ms", "pool by profile ($S^T$)"),
          ("phase_lr_incidence_ms", "incidence ($C$)"),
          ("phase_lr_query_ms", "type-major GEMM ($R$)"),
          ("phase_normalise_ms", "normalise"),
          ("phase_zeta_ms", "zeta block")]


def fig_phases(rows, P, outdir):
    ds = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, len(ds), figsize=(3.6 * len(ds), 4.0),
                             sharey=True, squeeze=False)
    axes = axes[0]
    keys = [k for k, _ in PHASES if any(r.get(k) for r in rows)]
    names = dict(PHASES)
    out_rows = []
    for ax, d in zip(axes, ds):
        Ts = sorted({r["T"] for r in rows if r["d"] == d})
        bottom = np.zeros(len(Ts))
        x = np.arange(len(Ts))
        for i, k in enumerate(keys):
            vals = np.array([next((r.get(k) or 0.0) for r in rows
                                  if r["d"] == d and r["T"] == t) for t in Ts])
            ax.bar(x, vals, bottom=bottom, width=0.68,
                   color=P["series"][i % 5], label=names[k],
                   edgecolor=P["surface"], linewidth=0.6)
            for t, v in zip(Ts, vals):
                out_rows.append((d, t, names[k], v))
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels([pow2(t) for t in Ts], rotation=45)
        ax.set_title(f"$d={d}$"); ax.set_xlabel("$T$")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("prefill time (ms)")
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Where prefill time goes", y=1.0)
    fig.text(0.0, -0.08,
             "For $d \\leq 3$ the scan dominates at every measured $T$: the "
             "long-range term is at most linear, which is the point of pooling.",
             fontsize=8, color=P["secondary"])
    save(fig, outdir, "fig3_phases")
    with open(os.path.join(outdir, "fig3_phases.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["d", "T", "phase", "ms"])
        w.writerows(out_rows)


def fig_decode(rows, P, outdir):
    ds = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    allT = sorted({r["T"] for r in rows})
    for i, d in enumerate(ds):
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        T, y = series(rows, d, "decode_us_per_token")
        axes[0].plot(T, y, color=c, marker=mk, label=f"d={d}")
        T, y = series(rows, d, "cache_words")
        axes[1].plot(T, y, color=c, marker=mk, label=f"d={d}  ($O(Brp)$)")
    T, y = series(rows, ds[0], "kv_cache_words")
    axes[1].plot(T, y, color=P["muted"], marker="x", ls="--",
                 label="softmax KV cache  ($O(T)$)")
    axes[0].set_yscale("log"); axes[0].set_ylabel("$\\mu$s per token")
    axes[0].set_title("Decode: cost per token is flat in $T$")
    axes[1].set_yscale("log"); axes[1].set_ylabel("cache (words)")
    axes[1].set_title("Decode: cache is $O(B\\,rp) = O(T^{1-1/d} rp)$")
    for ax in axes:
        set_T_axis(ax, allT); ax.set_xlabel("context length $T$"); ax.legend()
    save(fig, outdir, "fig4_decode")


def fig_triton(rows, P, outdir):
    rows = [r for r in rows if r.get("prefill_torch_ms")]
    if not rows:
        return
    ds = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    allT = sorted({r["T"] for r in rows})
    for i, d in enumerate(ds):
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        T, a = series(rows, d, "prefill_ms")
        _, b = series(rows, d, "prefill_torch_ms")
        axes[0].plot(T, b / a, color=c, marker=mk, label=f"d={d}")
        T, a = series(rows, d, "prefill_peak_MB")
        _, b = series(rows, d, "prefill_torch_peak_MB")
        axes[1].plot(T, a, color=c, marker=mk, label=f"d={d} fused")
        axes[1].plot(T, b, color=c, marker=mk, ls="--", alpha=0.75,
                     label=f"d={d} torch")
    axes[0].axhline(1.0, color=P["muted"], ls=":", lw=1.2)
    axes[0].set_ylabel("torch time / fused time")
    axes[0].set_title("Speedup of the fused kernels")
    axes[1].set_yscale("log"); axes[1].set_ylabel("peak prefill memory (MB)")
    axes[1].set_title("Peak activation memory")
    for ax in axes:
        set_T_axis(ax, allT); ax.set_xlabel("context length $T$")
        ax.legend(fontsize=8)
    fig.text(0.0, -0.06,
             "The torch path materialises the $(nb, n_c, c, c)$ score tile and "
             "the $(nb, B, \\mathrm{deg}, r, p)$ gather; the fused kernels keep both in "
             "registers, which is what Sec. 3.3 assumes.",
             fontsize=8, color=P["secondary"])
    save(fig, outdir, "fig8_triton")


def fig_subtractive(rows, P, outdir):
    rows = [r for r in rows if r.get("relerr_subtractive_fp32") is not None]
    if not rows:
        return
    ds = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    allT = sorted({r["T"] for r in rows})
    for i, d in enumerate(ds):
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        T, y = series(rows, d, "retained_fraction")
        axes[0].plot(T, y, color=c, marker=mk, label=f"d={d} measured")
        _, q = series(rows, d, "q")
        if len(q):
            axes[0].plot(T, 1.0 / np.maximum(q, 1), color=c, ls=":", alpha=0.8,
                         label=f"d={d}   $1/q$")
        for key, ls, tag in (("relerr_additive_fp32", "-", "additive"),
                             ("relerr_subtractive_fp32", "--", "subtractive")):
            T, y = series(rows, d, key)
            axes[1].plot(T, y, color=c, marker=mk, ls=ls, alpha=0.9,
                         label=f"d={d} {tag}")
    axes[0].set_yscale("log"); axes[0].set_ylabel("kept mass / subtracted mass")
    axes[0].set_title("The retained fraction is $\\Theta(1/q)$")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative error of the long-range normalizer")
    axes[1].set_title("What the cancellation costs (fp32 accumulation)")
    for ax in axes:
        set_T_axis(ax, allT); ax.set_xlabel("context length $T$")
        ax.legend(fontsize=7.5)
    fig.text(0.0, -0.06,
             "Remark 3.10 measured: each hyperplane carries $1/q$ of the profile "
             "mass, so the subtractive route forms a small quantity as a "
             "difference of large ones.", fontsize=8, color=P["secondary"])
    save(fig, outdir, "fig9_subtractive")


def fig_constants(sets, P, outdir, param, name, title, note):
    """Sensitivity to a constant that does not touch the exponent."""
    if len(sets) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    allT = sorted({r["T"] for rows in sets.values() for r in rows})
    for i, (val, rows) in enumerate(sorted(sets.items())):
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        d = rows[0]["d"]
        T, y = series(rows, d, "prefill_ms")
        a = fit(T, y)
        axes[0].plot(T, y, color=c, marker=mk,
                     label=f"{param}={val}" + (f"  ($\\alpha$={a[0]:.2f})" if a else ""))
        T, y = series(rows, d, "prefill_peak_MB")
        if len(y) and not np.isnan(y).all():
            axes[1].plot(T, y, color=c, marker=mk, label=f"{param}={val}")
    T, y = series(sorted(sets.items())[0][1], sorted(sets.items())[0][1][0]["d"],
                  "sdpa_ms")
    if len(T):
        axes[0].plot(T, y, color=P["muted"], marker="x", ls="--",
                     label="causal SDPA")
    axes[0].set_yscale("log"); axes[0].set_ylabel("prefill (ms)")
    axes[0].set_title("Prefill: the constant moves $T^*$, not the slope")
    axes[1].set_yscale("log"); axes[1].set_ylabel("peak prefill memory (MB)")
    axes[1].set_title("Peak activation memory")
    for ax in axes:
        ax.set_xscale("log"); set_T_axis(ax, allT)
        ax.set_xlabel("context length $T$"); ax.legend(fontsize=8)
    fig.suptitle(title, y=1.01)
    fig.text(0.0, -0.06, note, fontsize=8, color=P["secondary"])
    save(fig, outdir, name)


# ---------------------------------------------------------------------------
# theory figures
# ---------------------------------------------------------------------------

def fig_counting(rows, P, outdir):
    ds = sorted({r["d"] for r in rows if r["d"] >= 2})
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    allT = sorted({r["T"] for r in rows})
    labels = []
    for i, d in enumerate(ds):
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        T, y = series(rows, d, "I_prof")
        a = fit(T, y)
        ax.plot(T, y, color=c, marker=mk,
                label=f"$I_{{prof}}$ d={d}  ($\\alpha$={a[0]:.2f}, "
                      f"pred {2 - 3/d:.2f})" if a else f"$I_{{prof}}$ d={d}")
        labels.append((T[-1], y[-1], f"$I_{{prof}}$ d={d}", c))
        T, y = series(rows, d, "I_tok")
        ax.plot(T, y, color=c, marker=mk, ls="--", alpha=0.7)
        labels.append((T[-1], y[-1], f"$I_{{tok}}$ d={d}", c))
    Ta = np.array(allT, float)
    ax.plot(Ta, Ta ** 2, color=P["muted"], ls=":", lw=1.4)
    labels.append((Ta[-1], Ta[-1] ** 2, "dense $T^2$", P["muted"]))
    ax.set_xscale("log"); ax.set_yscale("log")
    set_T_axis(ax, allT)
    ax.set_xlabel("context length $T$"); ax.set_ylabel("incidences")
    ax.set_title("Table 1: the pooled and direct incidence counts")
    ax.legend(fontsize=7.5, loc="upper left")
    place_end_labels(ax, labels)
    fig.text(0.0, -0.06,
             "Solid: $nnz(C)$, what the schedule of Sec. 3.3 pays.  Dashed: "
             "$\\sum_j \\nu(prof(j))$, what a direct scatter pays.  The gap is "
             "$q=\\Theta(T^{1/d})$ and it is exact, not fitted.",
             fontsize=8, color=P["secondary"])
    save(fig, outdir, "fig5_counting")


def fig_vc(rows, P, outdir):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    styles = {"vc_exact": ("o", 1.0, "exact VC of $M^{(d)}$"),
              "vc_C": ("s", 0.85, "exact VC of the incidence $C$  ($+1$)"),
              "witness": ("^", 0.7, "constructive witness (lower bound)")}
    for i, level in enumerate(["vc_exact", "vc_C", "witness"]):
        sub = [r for r in rows if r.get("level") == level]
        if not sub:
            continue
        mk, al, lab = styles[level]
        xs = np.array([r["d"] for r in sub], float)
        ys = np.array([r["value"] + (1 if level == "vc_C" else 0) for r in sub],
                      float)
        jitter = (np.array([np.log2(max(2, r["T"])) for r in sub]) % 1) * 0.0
        ax.scatter(xs + jitter + (i - 1) * 0.08, ys, marker=mk, alpha=al, s=48,
                   color=P["series"][i % 5], label=lab, zorder=3,
                   edgecolors=P["surface"], linewidths=0.6)
    lim = [0.5, max(r["d"] for r in rows) + 0.5]
    ax.plot(lim, lim, color=P["muted"], ls="--", lw=1.3, zorder=1)
    ax.annotate("VC $= d$", xy=(lim[1], lim[1]), xytext=(-4, 6),
                textcoords="offset points", color=P["muted"], fontsize=8.5,
                ha="right")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("expressivity parameter $d$")
    ax.set_ylabel("measured VC dimension")
    ax.set_title("Theorem 3.1(iii), verified")
    ax.legend(fontsize=8, loc="upper left")
    fig.text(0.0, -0.14,
             "Deciding VC is LOGNP-complete, so the exact search on the full\n"
             "$T\\times T$ mask reaches only small $T$.  The witness is a certified\n"
             "lower bound at every $T$, and $VC(C) = d-1$ is the geometric\n"
             "ingredient the proof turns into $d$.",
             fontsize=8, color=P["secondary"], linespacing=1.5, va="top")
    save(fig, outdir, "fig6_vc")


def fig_density(rows, P, outdir):
    ds = sorted({r["d"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    allT = sorted({r["T"] for r in rows})
    labels = []
    for i, d in enumerate(ds):
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        T, y = series(rows, d, "density")
        ax.plot(T, y, color=c, marker=mk, label=f"$L_n$ block, d={d}")
        labels.append((T[-1], y[-1], f"d={d}", c))
        T, y = series(rows, d, "density_In_variant_P0")
        ax.plot(T, y, color=c, marker=mk, ls="--", alpha=0.7)
    ax.axhline(0.25, color=P["muted"], ls=":", lw=1.3)
    ax.annotate("two triangles: $1/4$", xy=(allT[0], 0.25), xytext=(2, 4),
                textcoords="offset points", color=P["muted"], fontsize=8)
    set_T_axis(ax, allT)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("context length $T$"); ax.set_ylabel("$nnz(M)/T^2$")
    ax.set_title("Theorem 3.1(iv): density with $L_n$ (solid) and $I_n$ (dashed)")
    ax.legend(fontsize=8)
    fig.text(0.0, -0.06,
             "Under $I_n$ the top-left block contributed $n$ nonzeros and the "
             "density fell to $\\approx1/8$, which is what the global set $P$ "
             "was carrying; under $L_n$ the two triangles alone give $1/4$.",
             fontsize=8, color=P["secondary"])
    save(fig, outdir, "fig7_density")


# ---------------------------------------------------------------------------
# routing figures
# ---------------------------------------------------------------------------

def _agg(rows, key, **sel):
    """Mean and min--max spread of ``key`` over seeds, keyed by k."""
    acc = defaultdict(list)
    for r in rows:
        if all(str(r.get(f)) == str(v) for f, v in sel.items()):
            if r.get(key) is not None:
                acc[int(r["k"])].append(float(r[key]))
    ks = sorted(acc)
    return (np.array(ks),
            np.array([np.mean(acc[k]) for k in ks]),
            np.array([np.min(acc[k]) for k in ks]),
            np.array([np.max(acc[k]) for k in ks]))


def _ceiling(rows) -> Dict[int, Dict[int, float]]:
    """``{d: {k: answerable_frac}}`` from the certified benchmark's rows."""
    out: Dict[int, Dict[int, float]] = defaultdict(dict)
    for r in rows or ():
        out[int(r["d"])][int(r["k"])] = float(r["answerable_frac"])
    return out


def fig_routing(rows, P, outdir):
    FLOOR = 1e-12                       # exact answers land at fp64 round-off
    ds = sorted({r["d"] for r in rows})
    have_task = any(r.get("nmse") is not None for r in rows)
    ncol = 3 if have_task else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.9 * ncol, 4.1))

    for i, d in enumerate(ds):
        sub = sorted((r for r in rows if r["d"] == d), key=lambda r: r["k"])
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        ks = [r["k"] for r in sub]
        axes[0].plot(ks, [r["n_patterns"] for r in sub], color=c, marker=mk,
                     label=f"d={d}")
        axes[0].plot(ks, [r["sauer"] for r in sub], color=c, ls=":", alpha=0.55)
        axes[1].plot(ks, [r["answerable_frac"] for r in sub], color=c,
                     marker=mk, label=f"d={d}")
        axes[1].axvline(d, color=c, ls="--", lw=0.9, alpha=0.45)
        if have_task:
            axes[2].plot(ks, [max(r["nmse"], FLOOR) for r in sub], color=c,
                         marker=mk, label=f"d={d}")
    ks_all = sorted({r["k"] for r in rows})
    axes[0].plot(ks_all, [2 ** k for k in ks_all], color=P["muted"], ls="--",
                 lw=1.3, label="$2^k$ (shattered)")

    axes[0].set_yscale("log")
    axes[0].set_ylabel("$\\Pi_M(k)$  (routing patterns)")
    axes[0].set_title("Shatter function on $k$ marked channels")
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].set_ylabel("fraction of requests $A$ answerable")
    axes[1].set_title("The transition at $k = \\mathrm{VC}(M) = d$")
    if have_task:
        axes[2].set_yscale("log")
        axes[2].axhline(FLOOR, color=P["muted"], ls=":", lw=1.2)
        axes[2].annotate("fp64 round-off floor", xy=(ks_all[0], FLOOR),
                         xytext=(2, 5), textcoords="offset points",
                         color=P["muted"], fontsize=8)
        axes[2].set_ylabel("normalised squared error")
        axes[2].set_title("$d$-Subset Routing, through the kernel")
    for ax in axes:
        ax.set_xlabel("requested routing dimension $k$")
        ax.set_xticks(ks_all)
        ax.legend(fontsize=8)
    fig.suptitle("$d$-Subset Routing: VC dimension as an operational capacity",
                 y=1.02)
    fig.text(0.0, -0.07,
             "Dotted: the Sauer--Shelah ceiling $\\sum_{i\\leq d}\\binom{k}{i}$ a "
             "VC-$d$ mask obeys.  Every request is answerable exactly while "
             "$k \\leq d$; past $d$ the mask cannot separate the requests, and "
             "the error is what no model carrying it can avoid.",
             fontsize=8, color=P["secondary"])
    dump(os.path.join(outdir, "fig12_routing.csv"),
         ["d", "k", "n_patterns", "sauer", "answerable_frac", "nmse"],
         [[r["d"] for r in rows], [r["k"] for r in rows],
          [r["n_patterns"] for r in rows], [r["sauer"] for r in rows],
          [r["answerable_frac"] for r in rows],
          [r.get("nmse") for r in rows]])
    save(fig, outdir, "fig12_routing")


def fig_routing_learned(rows, P, outdir, ceiling_rows=None):
    """The learned run against the certified ceiling.

    Three panels, answering three different objections:

      left    does SGD find the routes?  Learned NMSE per d, with the ceiling
              from bench_routing.py drawn under it.  Tracking it is the claim.
      middle  the transition at k = d, measured on a trained model rather than
              derived from the mask.
      right   causal influence.  Resample the payloads the request did NOT ask
              for and measure how far the prediction moves.  For SMAT at
              k <= d this is bit-exactly zero -- the mask removed those edges
              -- while a content-addressed baseline attends to them with small
              positive weight.  Unlike a leakage reading, this isolates the
              architecture from the fit.
    """
    FLOOR = 1e-6
    ds = sorted({r["d"] for r in rows if r["d"] > 0})
    ceiling = _ceiling(ceiling_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3))

    # -- left: learned error per d, against the certified ceiling ------------
    for i, d in enumerate(ds):
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        ks, mu, lo, hi = _agg(rows, "nmse", arm="smat", d=d)
        if not len(ks):
            continue
        axes[0].plot(ks, np.maximum(mu, FLOOR), color=c, marker=mk, label=f"d={d}")
        axes[0].fill_between(ks, np.maximum(lo, FLOOR), np.maximum(hi, FLOOR),
                             color=c, alpha=0.18, lw=0)
        cl = ceiling.get(d, {})
        if cl:
            axes[0].plot([k for k in ks if k in cl],
                         [max(1.0 - cl[k], FLOOR) for k in ks if k in cl],
                         color=c, ls=":", alpha=0.6, lw=1.2)
    ks_all = sorted({r["k"] for r in rows})
    for arm in ("softmax", "softmax_content"):
        ks, mu, _, _ = _agg(rows, "nmse", arm=arm)
        if len(ks):
            ls, mk, al = ARM_STYLE[arm]
            axes[0].plot(ks, np.maximum(mu, FLOOR), color=P["muted"], ls=ls,
                         marker=mk, alpha=al, label=arm)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("normalised squared error (fresh payloads)")
    axes[0].set_title("Learned error, with the certified ceiling")

    # -- middle: the transition, measured ------------------------------------
    for i, d in enumerate(ds):
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        ks, mu, lo, hi = _agg(rows, "exact_support_frac", arm="smat", d=d)
        if not len(ks):
            continue
        axes[1].plot(ks, mu, color=c, marker=mk, label=f"d={d}")
        axes[1].fill_between(ks, lo, hi, color=c, alpha=0.18, lw=0)
        cl = ceiling.get(d, {})
        if cl:
            axes[1].plot([k for k in ks if k in cl],
                         [cl[k] for k in ks if k in cl],
                         color=c, ls=":", alpha=0.6, lw=1.2)
        axes[1].axvline(d, color=c, ls="--", lw=0.8, alpha=0.35)
    for arm in ("softmax", "softmax_content"):
        ks, mu, _, _ = _agg(rows, "exact_support_frac", arm=arm)
        if len(ks):
            ls, mk, al = ARM_STYLE[arm]
            axes[1].plot(ks, mu, color=P["muted"], ls=ls, marker=mk, alpha=al,
                         label=arm)
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].set_ylabel("exact routing-pattern match")
    axes[1].set_title("The transition at $k=d$, after training")

    # -- right: causal influence of the unrequested payloads -----------------
    # Resampling an unrequested payload moves a content-addressed model's output
    # and cannot move SMAT's while k <= d, because the edge does not exist.
    FZ = 1e-9
    for i, d in enumerate(ds):
        c, mk = P["series"][i % 5], MARKERS[i % 5]
        ks, mu, lo, hi = _agg(rows, "influence_max", arm="smat", d=d)
        if not len(ks):
            continue
        axes[2].plot(ks, np.maximum(mu, FZ), color=c, marker=mk, label=f"d={d}")
        axes[2].fill_between(ks, np.maximum(lo, FZ), np.maximum(hi, FZ),
                             color=c, alpha=0.18, lw=0)
    for arm in ("softmax", "softmax_content"):
        ks, mu, _, _ = _agg(rows, "influence_max", arm=arm)
        if len(ks):
            ls, mk, al = ARM_STYLE[arm]
            axes[2].plot(ks, np.maximum(mu, FZ), color=P["muted"], ls=ls,
                         marker=mk, alpha=al, label=arm)
    axes[2].axhline(FZ, color=P["muted"], ls=":", lw=1.1)
    axes[2].annotate("exactly zero", xy=(ks_all[0], FZ), xytext=(2, 5),
                     textcoords="offset points", color=P["muted"], fontsize=8)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("max output change / payload scale")
    axes[2].set_title("Influence of the unrequested payloads")

    for ax in axes:
        ax.set_xlabel("requested routing dimension $k$")
        ax.set_xticks(ks_all)
        ax.legend(fontsize=7.5)
    fig.suptitle("$d$-Subset Routing, learned: SGD against the certified ceiling",
                 y=1.02)
    fig.text(0.0, -0.07,
             "Dotted, per $d$: the ceiling from bench_routing.py -- what the mask "
             "permits before any training.  Bands are min--max over seeds.  "
             "softmax_content is full causal attention that can address the "
             "request by content; it is the baseline the comparison has to "
             "survive, not one it should be spared.  Right-hand values plotted "
             "at the floor are exact zeros.",
             fontsize=8, color=P["secondary"])
    save(fig, outdir, "fig13_routing_learned")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default=None,
                    help="a bench_smat.py results CSV")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--results-dir", default="results",
                    help="where the jobs wrote their CSVs: the r/c sweeps, the "
                         "verify_theory tables, and the routing benchmarks")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args(argv)

    P = theme(args.dark)
    apply_style(P)
    os.makedirs(args.outdir, exist_ok=True)

    if args.results and os.path.exists(args.results):
        rows = load(args.results)
        print(f"timing figures from {args.results} ({len(rows)} rows):")
        fig_scaling(rows, P, args.outdir)
        fig_incidence(rows, P, args.outdir)
        fig_phases(rows, P, args.outdir)
        fig_decode(rows, P, args.outdir)
        fig_triton(rows, P, args.outdir)
        fig_subtractive(rows, P, args.outdir)
    else:
        print("no results CSV given; theory figures only")

    import glob
    for param, pat, name, title, note in (
            ("r", "results_r*.csv", "fig10_rank",
             "Sensitivity to the kernel feature rank $r$ (at $d=3$)",
             "The incidence phase costs $nnz(C)\\,r\\,p$, so $r$ scales every "
             "phase alike: it moves the crossover without touching the exponent."),
            ("c", "results_c*.csv", "fig11_chunk",
             "Sensitivity to the scan chunk width $c$ (at $d=3$)",
             "Peak memory is flat in $c$ because the fused kernel keeps the "
             "$c\\times c$ score tile in registers; under the torch path it "
             "would grow as $c^2$.")):
        sets = {}
        for path in sorted(glob.glob(os.path.join(args.results_dir, pat))):
            key = os.path.basename(path)[len("results_" + param):-len(".csv")]
            try:
                sets[int(key)] = load(path)
            except ValueError:
                continue
        if len(sets) >= 2:
            print(f"constant-sensitivity figure from {pat}:")
            fig_constants(sets, P, args.outdir, param, name, title, note)

    for name, fn in (("theory_counting.csv", fig_counting),
                     ("theory_vc.csv", fig_vc),
                     ("theory_structure.csv", fig_density)):
        path = os.path.join(args.results_dir, name)
        if os.path.exists(path):
            print(f"theory figure from {name}:")
            fn(load(path), P, args.outdir)
        else:
            print(f"  (skipping {name}: not found)")

    # The routing pair.  fig12 is drawn from the largest T available, since the
    # counts are a function of d alone; fig13 needs the ceiling that its own
    # marked sets came from, which is the T the learned run was trained at.
    ceilings = {}
    for path in sorted(glob.glob(os.path.join(args.results_dir, "routing_T*.csv"))):
        rows = load(path)
        if rows:
            ceilings[rows[0]["T"]] = rows
    if ceilings:
        # Prefer the largest T that was run with the task, not merely the
        # largest T: a --no-task table has no nmse column, and drawing fig12
        # from it would silently drop the third panel.
        with_task = [T for T, rs in ceilings.items()
                     if any(r.get("nmse") is not None for r in rs)]
        T = max(with_task) if with_task else max(ceilings)
        print(f"routing figure from routing_T{T}.csv:")
        fig_routing(ceilings[T], P, args.outdir)
    else:
        print("  (skipping routing_T*.csv: not found)")

    learned = os.path.join(args.results_dir, "routing_learned.csv")
    if os.path.exists(learned):
        rows = load(learned)
        T = rows[0]["T"] if rows else None
        print(f"learned routing figure from routing_learned.csv (T={T}):")
        fig_routing_learned(rows, P, args.outdir, ceilings.get(T))
        if T not in ceilings:
            print(f"  (no routing_T{T}.csv: drawn without the ceiling)")
    else:
        print("  (skipping routing_learned.csv: not found)")

    print(f"\nfigures in {args.outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

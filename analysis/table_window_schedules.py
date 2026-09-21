"""Tables for the boundary-schedule study (tasks/window_schedules.py).

  python analysis/table_window_schedules.py [results/window_schedules]
"""
import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "results/window_schedules"
ORDER = ["none", "half", "half_oracle", "win4", "win8", "dbl"]


KEYS = ["task", "sched", "trained", "seed", "T_eval", "layout", "D", "pos_lo"]


def order(df, col="sched"):
    df = df.copy()
    keys = [k for k in KEYS if k in df.columns]
    df = df.drop_duplicates(subset=keys, keep="last")      # preempted jobs restart and re-append
    df[col] = pd.Categorical(df[col], ORDER, ordered=True)
    return df


mq = os.path.join(root, "mqar.csv")
if os.path.exists(mq):
    d = order(pd.read_csv(mq))
    d["acc"] *= 100
    print("\n## MQAR, uniform layout: accuracy (%) by evaluation length, mean (sd) over seeds")
    u = d[d.layout == "uniform"].groupby(["sched", "T_eval"], observed=True)["acc"].agg(["mean", "std", "count"])
    print(u.apply(lambda r: f"{r['mean']:.1f} ({0 if pd.isna(r['std']) else r['std']:.1f}) n={int(r['count'])}", axis=1)
          .unstack("T_eval").to_string())
    print("\n## MQAR, placed layout: accuracy (%) by evaluation length and pair distance D (mean over seeds)")
    p = d[d.layout == "placed"].groupby(["T_eval", "D", "sched"], observed=True)["acc"].mean().unstack("sched")
    print(p.round(1).to_string())
    print("\n## share of query rows for which the whole pair region is behind the boundary (read through G)")
    g = d[d.layout == "placed"].groupby(["T_eval", "D", "sched"], observed=True)["pairs_in_G"].mean().unstack("sched")
    print(g.round(2).to_string())
    print("\ntraining:", d.groupby("trained", observed=False)[["steps", "train_sec"]].mean().round(0).to_string())

pg = os.path.join(root, "pg19.csv")
if os.path.exists(pg):
    d = order(pd.read_csv(pg))
    T = int(d.T_train.iloc[0])
    ranges = [(0, T // 2), (T // 2, T), (T, 2 * T), (2 * T, 4 * T), (4 * T, 8 * T)]
    rows = []
    for (sc, Te, seed), g in d.groupby(["sched", "T_eval", "seed"], observed=True):
        r = {"sched": sc, "T_eval": Te, "seed": seed}
        for a, b in ranges:
            s = g[(g.pos_lo >= a) & (g.pos_hi <= b)]
            if len(s):
                r[f"[{a},{b})"] = (s.bpb * (s.pos_hi - s.pos_lo)).sum() / (s.pos_hi - s.pos_lo).sum()
        rows.append(r)
    t = pd.DataFrame(rows)
    print("\n## PG-19 bytes: bits per byte by position range")
    print(t.round(4).to_string(index=False))
    # the bins just before and after each block boundary show the cost of a reset
    print("\n## PG-19 bytes: bits per byte in the bin before vs after each multiple of T/8 (reset bumps)")
    bw = int(d.pos_hi.iloc[0] - d.pos_lo.iloc[0])
    for (sc, Te), g in d.groupby(["sched", "T_eval"], observed=True):
        g = g.groupby("pos_lo")["bpb"].mean()
        step = T // 8
        diffs = []
        for x in range(2 * step, int(Te), step):
            if x - bw in g.index and x in g.index:
                diffs.append(g[x] - g[x - bw])
        if diffs:
            print(f"{sc:12s} T_eval={Te}: mean jump {sum(diffs) / len(diffs):+.4f} bpb over {len(diffs)} boundaries")

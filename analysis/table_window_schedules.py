"""Tables of the stepped-window appendix (tasks/window_schedules.py), as the paper prints them.

  python analysis/table_window_schedules.py [results/window_schedules]

  tab:mqar-uniform  MQAR accuracy (%) by evaluation length, uniform layout
  tab:mqar-placed   MQAR accuracy (%) by gap D at evaluation length 8192, placed
                    layout, and the share of query rows whose pair region is
                    read through G
  tab:pg19          PG-19 bits per byte by position within a 16384-byte window

MQAR is the hard variant (64 pairs over a noise vocabulary, every key twice):
mqar_hard.csv holds seeds 0 and 1, mqar_hard_s2.csv seed 2.  PG-19 seed 0 is in
pg19.csv and seeds 1 and 2 in pg19_s{1,2}*.csv, one file per schedule.  Every
cell is mean (sample std) over the three seeds.
"""
import glob
import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "results/window_schedules"
KEYS = ["task", "sched", "trained", "seed", "T_eval", "layout", "D", "pos_lo"]
ROWS = [("none", "No long-range branch"),
        ("half", "Boundary kept, current"),
        ("half_oracle", "Rebuilt, current"),
        ("dbl", "Doubling"),
        ("win8", r"Window, $c_{dc}=T_{\max}/8$"),
        ("win4", r"Window, $c_{dc}=T_{\max}/4$")]


def load(*patterns):
    fs = [f for p in patterns for f in sorted(glob.glob(os.path.join(root, p)))]
    d = pd.concat([pd.read_csv(f) for f in fs])
    # preempted jobs restart and re-append, so the last row of a key is the final one
    return d.drop_duplicates(subset=[k for k in KEYS if k in d.columns], keep="last")


def cell(v, fmt):
    return f"{v.mean():{fmt}} ({v.std():{fmt}})"


def table(header, lines):
    print(" & ".join(header) + r" \\")
    for name, cells in lines:
        print(name + " & " + " & ".join(cells) + r" \\")
    print()


mq = load("mqar_hard*.csv")
mq["acc"] *= 100

print("% tab:mqar-uniform")
u = mq[mq.layout == "uniform"]
Ts = (1024, 2048, 4096, 8192)
table(["Schedule", *map(str, Ts)],
      [(n, [cell(u[(u.sched == s) & (u.T_eval == T)].acc, ".1f") for T in Ts]) for s, n in ROWS])

print("% tab:mqar-placed")
p = mq[(mq.layout == "placed") & (mq.T_eval == 8192)]
Ds = (0, 256, 1024, 3072, 7678)
table(["Schedule", *(f"$D={D}$" for D in Ds)],
      [(n, [cell(p[(p.sched == s) & (p.D == D)].acc, ".1f") for D in Ds]) for s, n in ROWS])
table(["Share read through $G$", *(f"$D={D}$" for D in Ds)],
      [(n, [f"{p[(p.sched == s) & (p.D == D)].pairs_in_G.mean():.2f}" for D in Ds])
       for s, n in (("half", "Fixed boundary / doubling"), ("win4", r"Window, $c_{dc}=T_{\max}/4$"))])

print("% tab:pg19")
pg = load("pg19.csv", "pg19_s*.csv")
pg = pg[pg.T_eval == 16384]
bins = [(0, 1024), (2048, 4096), (4096, 8192), (8192, 16384)]
rows = []
for (s, seed), g in pg.groupby(["sched", "seed"]):
    r = {"sched": s, "seed": seed}
    for a, b in bins:
        x = g[(g.pos_lo >= a) & (g.pos_hi <= b)]
        r[(a, b)] = (x.bpb * (x.pos_hi - x.pos_lo)).sum() / (x.pos_hi - x.pos_lo).sum()
    rows.append(r)
t = pd.DataFrame(rows)
table(["Schedule", *(f"$[{a},{b})$" for a, b in bins)],
      [(n, [cell(t[t.sched == s][c], ".3f") for c in bins]) for s, n in ROWS])

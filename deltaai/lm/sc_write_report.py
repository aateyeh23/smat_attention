"""Summarize the matched write-gate runs and historical Mamba-2 learning curves."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()
delta = Path(__file__).resolve().parents[1]
paths = [(p, "independent" if "independent" in p.stem else "shared")
         for p in sorted(args.run_dir.glob("reset-*.log"))]
paths += [(delta / "logs/sc4k-nc16-ds16-mamba2-d1.out", "historical_mamba2")]
paths += [(p, "historical_mamba2") for p in sorted((delta / "logs").glob("sc4k-tweak-d3-goff*.out"))]
rows = []; statuses = {}; trajectories = {}
for path, arm in paths:
    source = path.read_text()
    halves = None
    trajectory = {}
    for line in source.splitlines():
        match = re.match(r"SOURCE_HALF first=([\d.]+) second=([\d.]+)", line)
        if match:
            halves = [float(value) for value in match.groups()]
        match = re.match(r"EVAL step (\d+) acc ([\d.]+)", line)
        if match:
            step, acc = int(match[1]), float(match[2])
            row = dict(arm=arm, run=path.stem, step=step, accuracy=acc,
                       first_half=halves[0] if halves else None, second_half=halves[1] if halves else None)
            rows.append(row); trajectory[step] = row
    trajectories.setdefault(arm, []).append(trajectory)
    if arm != "historical_mamba2":
        statuses[path.stem] = dict(
            latest_eval=max(trajectory) if trajectory else None,
            complete=bool(re.search(r"^FINAL step 10000 |^SOLVED at step ", source, re.M)),
            error="Traceback (most recent call last)" in source,
        )

common = set.intersection(*(set(run) for runs in trajectories.values() for run in runs))
step = max(common) if common else None
summary = {"matched_step": step, "runs": statuses, "arms": {}}
if step is not None:
    for arm, runs in trajectories.items():
        values = [run[step]["accuracy"] for run in runs]
        summary["arms"][arm] = dict(mean=statistics.mean(values), values=values)
        for field in ("first_half", "second_half"):
            parts = [run[step][field] for run in runs]
            if all(value is not None for value in parts):
                summary["arms"][arm][field] = statistics.mean(parts)
with (args.run_dir / "learning_curves.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["arm", "run", "step", "accuracy", "first_half", "second_half"])
    writer.writeheader(); writer.writerows(rows)
(args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))

if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = {"independent": "Reset SMAT: independent writes (3 runs)",
              "shared": "Reset SMAT: shared writes (3 runs)",
              "historical_mamba2": "Mamba-2: historical (4 runs)"}
    colors = {"independent": "#1769aa", "shared": "#d97706", "historical_mamba2": "#606060"}
    for axis, field, title in zip(axes, ["accuracy", "first_half"], ["All answer tokens", "Tokens from before the reset"]):
        for arm, runs in trajectories.items():
            steps = sorted(set.intersection(*(set(run) for run in runs)))
            steps = [s for s in steps if s <= step and all(run[s][field] is not None for run in runs)]
            if not steps:
                continue
            values = [[100 * run[s][field] for run in runs] for s in steps]
            axis.plot(steps, [statistics.mean(v) for v in values], label=labels[arm], color=colors[arm])
            axis.fill_between(steps, [min(v) for v in values], [max(v) for v in values], color=colors[arm], alpha=.12)
        axis.set(title=title, xlabel="Training step", ylabel="Token accuracy (%)", ylim=(0, 100))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Selective copying · L=4096 · 16 tokens · bands show run min–max")
    fig.savefig(args.run_dir / "learning_curves.png", dpi=180)

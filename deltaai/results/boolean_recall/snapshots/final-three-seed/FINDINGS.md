# Retrieval plus conjunction: completed three-seed experiment

**The controls learn, but SMat d3 does not show a consistent advantage.** All
models solve supplied-bit AND perfectly in all three seeds. Retrieval is less
reliable, and both SMat variants have lower mean two-bit retrieval accuracy
than their native backbones. End-to-end AND is nearly tied for GDN on average,
with a substantial SMat win in one seed and losses in the other two.

Completed36 runs: three conditions × four models × three seeds123/456/789.
Every run uses four records, T64, width64, two layers, and exactly500 steps on
DeltaAI `ghx4-interactive`. The initial12-run screen was archived before adding
seeds. No task, model, loss or optimizer changes were made during confirmation.

## Main results

Mean ± sample SD across three seeds, in percent. Retrieval exact accuracy
requires both requested bits correct; end-to-end AND requires one answer.
These are different success criteria, not directly comparable error rates.

| Model | Supplied bits → AND | Retrieve both bits exactly | Retrieve + AND |
|---|---:|---:|---:|
| Mamba-2 | 100.00 ± 0.00 | 59.81 ± 14.96 | 83.54 ± 2.25 |
| Mamba-2 + SMat d3 | 100.00 ± 0.00 | 46.20 ± 7.88 | 78.88 ± 3.43 |
| GDN | 100.00 ± 0.00 | 87.00 ± 9.75 | 88.53 ± 2.71 |
| GDN + SMat d3 | 100.00 ± 0.00 | 67.55 ± 15.09 | 87.81 ± 6.18 |

All four models score100% on every direct-AND truth case, not just overall.
The end-to-end AND paired SMat gain is −4.65 ± 5.53 percentage points for
Mamba and −0.72 ± 8.22 points for GDN. These small samples do not establish a
population-level difference or equivalence.

| Seed | Mamba AND | Mamba+SMat AND | GDN AND | GDN+SMat AND |
|---|---:|---:|---:|---:|
| 123 | 85.99 | 74.95 | 91.58 | 84.45 |
| 456 | 81.57 | 80.42 | 87.60 | 84.03 |
| 789 | 83.06 | 81.27 | 86.40 | 94.95 |

GDN+SMat d3's94.95% seed789 result is a real successful run (+8.54 points over
its paired backbone), but the other two paired gains are −7.13 and −3.56 points.
Mamba+SMat loses all three paired AND comparisons. No larger-load sweep was
launched on the basis of the isolated GDN success.

## Shortcuts and truth-table results

Every test contains exactly1024 examples each of00/01/10/11. Always answering0
scores75%. AND accuracy alone therefore overstates successful retrieval.
Mean11-case accuracy is57.13% for Mamba,28.87% for Mamba+SMat,70.41% for GDN,
and66.83% for GDN+SMat. The full four-case tables are in [report.md](report.md).

A privileged oracle knowing only memory-wide bit counts reaches81.21 ± 0.54%
on these test sets. An oracle knowing those counts plus only one queried bit
reaches roughly87.7%; exact per-seed references are in
[shortcut_controls.json](shortcut_controls.json). These are diagnostic
references, not trained baselines or universal bounds. The models' overall
scores alone do not prove that they explicitly retrieve both queried bits.

Query-specificity provides another check: correct AND accuracy minus accuracy
against other key pairs' AND outcomes averages2.96 ± 2.89pp for Mamba,
0.04 ± 0.77pp for Mamba+SMat,10.51 ± 3.58pp for GDN, and9.70 ± 9.53pp for
GDN+SMat. This supports very weak queried-pair discrimination for Mamba+SMat
in these runs. It does not identify the internal retrieval algorithm.

## What the controls tell us

Local AND computation is learnable by every tested model within500 steps.
The retrieval-only condition already exhibits seed variation and lower mean
SMat performance. Thus conjunction is not required to observe that weakness.

This does not uniquely isolate a causal bottleneck. The three conditions train
separate models, and retrieval supervises two labels rather than one. For
example, applying exact AND externally to native GDN's retrieved bits gives
94.21 ± 4.42%, compared with88.53 ± 2.71% when trained on the AND answer alone.
For GDN+SMat the corresponding numbers are84.42 ± 7.66% and87.81 ± 6.18%.
The objective changes what gets learned; these comparisons do not show that
the internal conjunction operation alone accounts for the difference.

We have established a learnable component-control experiment, not a mask-only
ablation, a capacity ceiling, or a robust SMat win. Existing SMat variants retain
their extra memory, parameters, routing loss and local-state reset. Only d3 was
tested here. Three seeds and four records limit generalization of the result.

## Verification and artifacts

[Audit](audit.json) passed for all36 runs: complete planned grid, independent
token/target decoding, equal truth-table counts, regenerated validation/test
hashes, recomputed accuracy/exact/truth-case/composed-AND/binding/control metrics,
paired memories and bit pairs across conditions, and matched recipes. Each
run's independent final test has4096 examples; validation has1024.

[Protocol](PROTOCOL.md), [full tables and plots](report.md), [CSV](results.csv),
[job accounting](jobs.txt). Jobs3172006 and3172032 completed successfully in
3m55s and7m05s, each on one interactive GPU. Source snapshots and hash manifests
preserve the implementation; the task, trainer and model sources remained
unchanged throughout. Per-run checkpoints and raw predictions are retained.

![Three-seed component comparison](w64-r4-t64.png)

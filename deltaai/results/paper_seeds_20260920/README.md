# Tables 2–3: additional training seeds on DeltaAI

Requested: four additional seeds for every configuration in Tables 2 and 3,
and five seeds for Log-Linear joint context–key recall. Both GDN and Mamba-2
Log-Linear baselines are included.

- New seeds: 1, 2, 3, 4; existing seed: 123.
- Table 2: 30 configurations × four seeds = 120 planned runs.
- Table 3: eight configurations × four seeds = 32 runs.
- Joint Log-Linear: two configurations × five seeds = ten runs.
- Every run uses 32 epochs. No accuracy-dependent stopping or seed selection.
- All training executes on DeltaAI GH200 GPUs. Historical Modal artifacts are
  read only to trace the original recipes/results.

`tasks.json` specifies every new run; `original_runs.json` identifies verified
completed seed-123 results. `source/` freezes the experiment implementations;
`source_sha256.json` verifies those files before dispatch. Model seeds change,
while datasets, widths, learning rates, optimizer, and evaluation definitions
retain the original recipes. MQAR uses the original fixed whole-batch ordering;
joint recall uses its original seed-dependent batch ordering. Consequently
MQAR variability measures initialization/dropout variability on fixed data/order.

Table 2 GDN width 16 uses the original per-profile delta memory. Widths 32 and
64 use the selected boundary-transport implementation, with scalar decay and
without incidence rescaling. Mamba uses one hard write/four reads, with its
original additive memory. The original per-length router allocation is retained.
Table 2 uses FP32 training; Table 3 uses BF16 autocast as before.

Eight worker shards resume durable checkpoints across two-hour Slurm allocations.
After the 152 table repeats finished, the two seed-4 joint Log-Linear runs
were assigned dedicated workers to avoid waiting behind earlier seeds. Those
tasks are excluded from the original shard dispatch; per-run file locks
prevent concurrent writes. The training recipes remain unchanged.
`status-*.json` records each shard, `logs/` contains job/training output. The
initial shared-filesystem Triton cache caused stale-file errors; subsequent
launches use `/tmp/$USER-paper-seeds-triton` on each compute node.

The working-tree Table 2 Log-Linear folder and Modal copy contain unfinished
runs. Completed SCF results were found in remote commit
`3265a5df793e2275ec56bbd9977b4be7993eb5ef` and copied to `original_loglinear/`.
All six accuracies match Table 2, and operator/data/interleaving source files
match the frozen repeat sources byte for byte. `mqar_loglinear_recipe_verified.json`
records this check and releases those 24 repeats. Existing source files and
the dirty working tree were preserved; the remote commit was not merged.
Joint Log-Linear runs are gated on `loglinear_validated.json`; the extension
uses the same dense binary-WEAK operator as Table 2, 13 shared levels, and the
native control projections. Batchwise activation checkpointing bounds matrix
storage at long lengths while preserving the full optimizer batch and outer
model dropout draws. BF16 autocast surrounds an FP32 dense core and q/k
normalization. Tests cover forward/gradient agreement, all five lengths through
3076, full training-batch memory/finite gradients, and causality. This is an
accuracy comparison, not an upstream-kernel throughput claim. An exploratory
upstream-kernel port failed compatibility checks and is retained only as
`joint_loglinear_upstream_prototype.py`; no training uses it.

Run `python deltaai/paper_seeds/collect.py` from the repository root to refresh
`report.md`, `summary.csv`, `summary.json`, and `per_seed.csv`. These use final
epoch metrics only and never treat incomplete runs as completed seeds. Means
and sample standard deviations are across training seeds. Table 3 validation
and final test remain separate. Original GDN width64 d2 raw accuracy is
89.6834375%, differing from the draft's 89.71%.

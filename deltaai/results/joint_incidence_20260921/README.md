# Joint context–key recall: incidence ablations

This campaign tests whether finite-geometric incidence improves learned recall
relative to equally sized random overlapping summaries. The primary comparison
changes only the incidence buffer, while retaining the learned writer, four-read
selector, GDN transport and reset, all parameter shapes, optimizer, data, and
training exposure.

There are 15 fresh runs: three training seeds (123, 456, 789), each with geometric
SMAT, random balanced partitions with topology seeds 17 and 29, direct independent
buckets with the same number of profiles, and a larger independent-bucket control
with the same profile-plus-summary matrix-table byte budget. Random draws are
crossed with training seeds; they are not six independent training seeds.

All models use GDN, d=3 geometry as the reference, width 64, two layers, two
heads with state/value dimensions 16, LR .003, AdamW decay .1, batch 256, and
32 full epochs. No run is stopped for poor accuracy. Each run sees 180,000
examples per epoch and takes 22,560 optimizer steps. The existing five-size
mixture contains 4, 16, 128, 256 and 512 bindings. The primary metric is final
test accuracy at 128 bindings; every size and macro accuracy are reported.

The profile-matched bucket control retains the exact writer and reads four
individual profile states. Its reader is smaller, and its table byte usage is
lower. The byte-matched control uses q(2q+1) states against geometric q^2 profiles
plus q(q+1) summaries, with the same two write projections and coordinate hash
but radices (q, 2q+1). It has more reader parameters. These are explicitly
labeled resource controls, not additional incidence-only comparisons.

Matrix-table accounting is FP32 payload storage per example across heads.
It is not a claim about an implemented autoregressive cache. Per-run ablation
metadata records profile count, read-type count, incidence edges, hash radices,
profile/summary/read table bytes, parameter counts, and topology digests. GPU
validation additionally measures the full 256-example, length-3076 training
batch. Validation process peak allocation includes the retained reference model
and checking tensors; it is not a dedicated inference memory benchmark.

Training uses a frozen copy of the previously confirmed source_v5 trainer and
models, plus a frozen Zoology package. New adapter and launcher files are covered
by the source SHA256 manifest. The existing cached dataset is shared read-only
through symlinks and checked against its original file hashes before training.
The adapters are applied before constructing the optimizer and reconstructible
on resume. Checkpoints and logs are isolated from all earlier experiments.

Validation job: 3188900 (passed). Training array: 3188913, four single-GPU
shards. The queued four-GPU job 3188903 was canceled before starting because
the scheduler estimated a substantially longer wait for a whole node.
The shard manifests partition the identical 15 tasks without duplication;
`scheduler_sha256.json` separately locks this scheduling-only change.
Initial validation 3188880 passed; initial training launch 3188884
failed before starting a worker because the historical snapshot lacked the
campaign launcher. The unchanged launcher was added and validation repeated;
`launch_fix.json` records the correction. Training uses up to four GPUs,
checkpoints on every epoch and at the time limit, and resumes unfinished runs
in at most three four-hour allocations per shard. A failed subprocess prevents further dispatch and
automatic continuation. A normal run is expected to need much less than the cap.

- `protocol.json`: fixed data, recipe, arms, seeds and metrics.
- `resources.md`: parameter counts, measured step times and memory-table sizes.
- `tasks.json`: all 15 commands and output locations.
- `validated.json`: source/data integrity and GPU forward/backward results.
- `shards/*/campaign_status.json`: current training jobs, completed tasks and failures.
- `logs/`: validation, allocation and individual training logs.
- `report.md` and `summary.json`: collected results; only full-budget runs count.

Refresh the report with
`python /u/archerdw/smat_attention/deltaai/incidence_ablation/collect.py`.
It is also refreshed automatically at the end of each allocation.

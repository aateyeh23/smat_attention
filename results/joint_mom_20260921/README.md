# MoM memory-budget comparison

Six fresh runs: `mom_profiles` and `mom_bytes`, each with seeds 123, 456, 789.
The comparison uses the original cached shared-key joint recall data and frozen
32-epoch trainer. Primary metric is final-test accuracy at 128 bindings. Report
all five loads, macro accuracy and exact recall, without test-based selection.

Each SMAT mixer is replaced by a MoM Gated DeltaNet mixer with top-4 weighted
routing, four updates and four reads per token, and no shared memory. Q/output
projections are shared across memories; K/V and beta/decay projections are
memory-specific. The two heads share the MoM router. The embedding, residual
normalizations, tied output head, width 64 and two-layer structure are retained.
Unlike SMAT, MoM processes the full causal sequence without a halfway reset.

For each sequence length, the number of independent states per head/layer is:

- `mom_profiles`: SMAT's profile count, q² (169 at length 3076).
- `mom_bytes`: SMAT's profile plus derived-summary count, q²+q(q+1)
  (351 at length 3076).

Both use 16×16 matrix states. At length 3076 the provisioned FP32 matrix-state
budget across two heads and two layers is 692,224 or 1,437,696 bytes per example.
This does **not** match independent information capacity, parameter counts,
compute, or total recurrent cache. MoM also has separate convolution histories
for every memory stream; these are accounted for separately in `ablation.json`.
The expert bank is shared across lengths using prefixes; routers are specific
to each length. This is a top-4/no-shared modification of MoM, not its default
top-2-plus-shared configuration, and is a whole-mixer baseline rather than an
incidence-only ablation.

The implementation snapshots installed FLA `MomAttention` for parameterization
and initialization. A packed forward uses stable chronological sorting,
segmented short convolution and batched memory-specific projections. Batch
shards of 32 with activation checkpointing bound temporary allocations while
keeping the effective batch 256 and optimizer steps unchanged. Each shard
contains complete examples. Validation compares forward and backward to an
independent explicit recurrent implementation, checks causality and batch
isolation, and tests training at every length plus batch 256 at maximum length.

Load balancing uses coefficient .01 times the mean per-layer Switch/MoM loss,
including all four route positions. Training otherwise uses the same fixed
recipe as the existing SMAT controls, without MoM-specific tuning.

Provenance is in `protocol.json` and `source_sha256.json`. GPU validation must
write a matching `validated.json` before training may run. Training writes
per-seed `status.json`, `metrics.jsonl`, resumable checkpoints and `result.json`.
`report.md`/`summary.json` are refreshed at allocation completion. Allocation
slices are four hours, checkpoint after 225 minutes, at most three slices/run.

Sources: https://arxiv.org/abs/2502.13685 and
https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mom.py.

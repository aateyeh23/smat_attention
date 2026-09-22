# Revised PG19 pretraining: width768, rank64 readers

**Campaign decision, September 17:** d=4 is excluded at the user's request.
The launcher now supports d=2/3 for GDN and Mamba-2 and rejects direct d=4
runs. Historical d=4 pilot results and checkpoints below are retained.
The d=2 correctness checks and four-step GPU pilots passed. No full
pretraining continuation has been launched. d=2 uses four point-summary reads
from 97 buckets at 16K; d=3 uses four hyperplane-summary reads from 552 types.
The collector defaults to d=2/3; use `--dimensions 3 4` for historical results.

Original pilot arms: GDN and Mamba-2, each with SMAT d=3/d=4; seed123.
All use16layers, residual width768, FFN2048, GPT2 vocabulary50257,
tied input/output embeddings and16K context. GDN uses6heads with key/value128;
Mamba-2 uses24heads, head/state64, native expansion2. Both have98304 recurrent
state elements per layer. No newly trained unaugmented baseline in this campaign.

Each SMAT reader has a shared learned768->64 projection within its layer and
independent per-head64->type scoring matrices. Thus each effective scoring matrix
has rank at most64. Queries retain4weighted distinct reads, keys1hard bucket write.
Both use the neighbor write-hash surrogate gradient, detached writer hash inputs,
no planted anchor writes, and no incidence output rescaling. GDN uses boundary
transport with scalar decay; Mamba uses scalar boundary decay and an independent
sigmoid write gate initialized at0.1. Local convolutions/recurrence reset at the
midpoint. Hash inputs use a learned causal convolution.

The optimizer/data recipe follows previous PG19 runs: LR3e-4,40M-token warmup,
cosine schedule, AdamW betas(0.9,0.95), weight decay0.1, global batch8sequences,
gradient clipping1, BF16 autocast. Initial pilots use microbatch1 and activation
checkpointing. Full-model pilots take4optimizer steps, preserving checkpoint,
optimizer,RNG,data cursor and original2B-token schedule for exact continuation.
The11.8-hour cap and2B-token target are both recorded; a2B completion time must
be assessed from measured throughput before launching continuation.

GPU gates compare factorized fused routing and gradients to an explicitly
materialized scoring matrix; verify task-loss gradients reach writer hashes;
check all trainable gradients and causal inference. No CPU model checks.
Loaded custom source paths and SHA256 are audited and copied to the volume.

Modal app/volume:pg19-w768-rank64; paths:/seed123/{gdn,mamba2}-d{3,4}.
Data:smat-pg19-scale-data-v1:/pg19-16k-2b.
Checkpoints every10minutes and at250M-token milestones; validation every50M.
SIGUSR1/SIGTERM or a STOP file requests a checkpoint and exit after an optimizer
step; use only after training initialization. Remove STOP before continuation.

Launch pilots: modal run --detach deltaai/modal_pg19_rank64.py::sweep
Continue: modal run --detach deltaai/modal_pg19_rank64.py::sweep --no-pilot
Collect: python deltaai/collect_pg19_rank64.py --watch

The earlier1024-wide, two-head GDN pilots are stopped and retain checkpoints on
pg19-updated-transport-d34. They are distinct experiments and are not resumed here.
Frozen base environment/data artifacts are external to Git.

## Fused write-gradient throughput candidate

The reference four-step pilots use microbatch1 with activation checkpointing and
the Python gather-based neighbor-write gradient. A separate candidate under
/seed123/fusedwrite-b2-ac0 uses a numerically verified fused Triton neighbor-write
gradient, microbatch2, and no activation checkpointing. It preserves the model,
training examples, global batch8, optimizer, and schedule. Its checkpoints and
source manifests are separate. The default launcher now selects this candidate.

The launcher selects microbatch2 for both d2 arms and GDN d3, and microbatch1 for Mamba d3;
all retain global batch8 through gradient accumulation. GDN d4 and both Mamba
arms exceeded device memory at microbatch2 without activation checkpointing.
The Triton compilation cache is container-local to avoid volume I/O during
autotuning. Sweep failures are collected independently so one failed arm does
not cancel the others.

Completed GDN pilots measured20,197tokens/s for d3 (microbatch2) and12,858tokens/s
for d4 (microbatch1), averaging steps2–4. These imply27.5/43.2hours for2Btokens
before evaluation and checkpoint overhead. Both saved524,288-token checkpoints.
Mamba d3/d4 also completed at microbatch1, saving524,288-token checkpoints.
They measured38,373/20,630tokens/s, respectively (14.5/26.9hours for2Btokens).
Their peak allocated GPU memory was45.4/65.4GB. All four pilots are stopped.
The main campaign is pending a choice between more GPUs and fewer tokens;
these short pilots do not establish language-model quality or long-run stability.

## d=2 pilots (September 17)

Both model families passed the factorized-reader output/gradient comparison,
finite full-model gradients, task-loss gradients to the write hash, and causal
inference checks on GPU. The fused write-gradient numerical checks now cover
two neighbor routes at state dimensions 16, 64 and 128, in addition to the
previous four/eight-route cases. Full 16-layer, width-768, 16K training pilots
completed four optimizer steps using the same global batch of eight sequences.

| Model | Trainable parameters | Microbatch | Mean tokens/s, steps 2–4 | Peak allocated GB | Hours / 500M | Hours / 2B |
|---|---:|---:|---:|---:|---:|---:|
| GDN + SMAT d2 | 163,191,968 | 2 | 24,643 | 59.44 | 5.64 | 22.54 |
| Mamba-2 + SMAT d2 | 176,913,280 | 1 | 54,307 | 35.32 | 2.56 | 10.23 |
| Mamba-2 + SMAT d2 | 176,913,280 | 2 | 58,745 | 67.92 | 2.36 | 9.46 |

Both d2 launch defaults select microbatch2. Relative to the earlier d3 pilots,
GDN throughput improves 22.0% at the same microbatch2; Mamba throughput improves
41.5% at the same microbatch1, or 53.1% with microbatch2. These are short pilot
comparisons, not repeated performance studies. Time projections exclude startup,
evaluation and checkpoint overhead; the first GDN step took 250 seconds and is
excluded from the steady-throughput average. No validation perplexity is available
at this training stage.

Each pilot saved a resumable checkpoint after 524,288 tokens. Selected checkpoints
are on Modal volume `pg19-w768-rank64` at
`/seed123/fusedwrite-b2-ac0/{gdn,mamba2}-d2/latest.pt`;
`d2-checkpoints.json` records remote file sizes and validation/status checks.
GDN app: `ap-OhHLXih33CbnpJ3cwEbu2u`; Mamba microbatch1:
`ap-4xX7HPcHGUWfpUBqAaSREt`; Mamba microbatch2:
`ap-e9bHOyJvyVwaOSy9ddMfD4`.

The d2 source updates extend model dimension guards and GPU tests. Existing d3
snapshots keep their original source manifests; the launcher deliberately rejects
resuming those snapshots with changed source hashes. A later d3 continuation must
use its frozen sources or explicitly validate and record a source migration.

The fused kernel computes each neighbor gradient as k^T (dL/dMemory_cell) v
without materializing gathered state matrices. GPU checks cover state sizes
16/64/128,4/8neighbor routes and strided inputs, followed by end-to-end task-loss
write-hash gradients and full-model pilots. The old gradient implementation
remains the default outside this explicitly configured campaign.

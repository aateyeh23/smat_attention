# Revised PG19 pretraining: width768, rank64 readers

Four arms: GDN and Mamba-2, each with SMAT d=3/d=4; seed123.
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

The fused kernel computes each neighbor gradient as k^T (dL/dMemory_cell) v
without materializing gathered state matrices. GPU checks cover state sizes
16/64/128,4/8neighbor routes and strided inputs, followed by end-to-end task-loss
write-hash gradients and full-model pilots. The old gradient implementation
remains the default outside this explicitly configured campaign.

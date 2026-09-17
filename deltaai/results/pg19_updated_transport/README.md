# PG19 updated transport: d=3 and d=4

Fresh seed123 runs, not resumed old delta-memory checkpoints. Model:16layers,
width1024, FFN2816,2heads, head/state64,16K context, GPT2 vocabulary. Target2B
unique prepared PG19 tokens;11.8-hour training cap; LR3e-4 with40M-token warmup,
cosine schedule, AdamW betas(0.9,0.95), weight decay0.1, gradient clipping1.0.
Global batch8 sequences, microbatches3/3/2, BF16 autocast, activation checkpointing.

The updated MQAR configuration uses GDN boundary transport,1hard bucket write,
4weighted hyperplane reads, scalar transport decay on, incidence rescaling off,
detached write-hash inputs, neighbor write-hash surrogate gradients on, planted
anchor writes off. This is transport-additive profile memory, not the previous
profile-delta update. Reader/router use existing Triton kernels; transport and
local recurrence use FLA kernels; profile aggregation uses cuBLAS.

Data volume:smat-pg19-scale-data-v1, /pg19-16k-2b.
New checkpoint volume:pg19-updated-transport-d34, /seed123/d3 and /seed123/d4.
Resumable state every10minutes; model milestones every250M tokens; held-out
validation every50M tokens; training metrics every10steps. All loaded custom
sources have paths/SHA256 audited and snapshots saved alongside checkpoints.
Full-context GPU validation precedes training for each d.

Launch: modal run --detach deltaai/modal_pg19_transport.py::sweep
The Modal image reuses the local frozen bundle from earlier MQAR runs; the bundle
and data/checkpoint volumes are external artifacts and are not stored in Git.

# MQAR Log-Linear sweep, seed123

Exactly six training arms: GDN and Mamba-2, widths16/32/64. LR0.01 except
GDN64 LR0.003. No new control/baseline training runs. Two layers, head and state
dimensions16; GDN heads1/2/2 and Mamba-2 expand2 heads2/4/8. Existing frozen
MQAR mixture, interleaved batches,32epochs, AdamW weight decay0.1, cosine schedule.
Checkpoints saved every epoch, no accuracy/relative-baseline stopping gates.

## Implementation and limits

Official reference: https://github.com/HanGuo97/log-linear-attention
commit7f8644159c1406fae1ad863829a5b3a4fbf63022.
The operator is base2 WEAK Log-Linear Attention with positive query-dependent
lambda=softplus(L*l_proj(u)), L initialized to1. A single9-level projection covers
all MQAR lengths64/128/256; unlike the upstream LM's15-level projection, it does
not allocate unused levels for lengths up to16K. Native Zoology control backbone
projections/convolutions/norms and initializations are retained; this is not the
full upstream Transformers LM configuration. Lambda projection is separate for
both families (upstream Mamba concatenates its columns in in_proj).

Execution uses dense PyTorch GPU operations, NOT upstream hierarchical Triton
kernels, and has quadratic storage. GDN uses a triangular solve mathematically
equivalent to upstream base.py hattention_materialized_dplr_v2; Mamba uses the
scalar-gated attention matrix in hattention_materialized_v2. GDN's upstream
SCALE_LAMBDA=True cancels q's1/sqrt(head_dim) output scaling. FLA q/k normalization
uses sum-of-squares+1e-6. All gradients flow through gates, lambdas, and transitions.
This is an accuracy experiment at short sequence lengths, not a measurement of
Log-Linear's asymptotic runtime or upstream kernel speed.

GPU gate checks outputs against official materialized references, outputs and
all input gradients against independent direct recurrence, and complete-mixer
finite forward/backward and causality for all six shapes. No CPU model tests and
no optimizer/training steps in this validation. Source SHA is checked at training
import; source snapshots accompany checkpoints in the Modal volume.

## Operation

Modal app: mqar-log-linear-s123; volume: mqar-log-linear-s123.
Each arm is /{gdn,mamba2}-w{16,32,64}; checkpoint history under checkpoints/.
Local launch log: launch.log; function IDs: campaign.json.
Collect live status and download final checkpoints:

    ~/venvs/modal-mqar/bin/python collect_log_linear.py --watch

Native baseline comparisons are from prior runs; the correct GDN64 LR0.003
baseline is trial30_w64_lr003_gdn, not the older LR0.01 baseline.

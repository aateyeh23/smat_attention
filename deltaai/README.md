# DeltaAI job scripts

Ports of the `../smat/*.sbatch` scripts for NCSA DeltaAI (GH200, partition
`ghx4`, account `bekw-dtai-gh`).  The Python code in `../smat` is used
unmodified; only the Slurm headers and environment differ.  Results and logs
are written here, not into the repo's `results/` and `logs/`.

    sbatch smoke_gpu.sbatch     # ~35 min check of the triton kernels

Notes on the port:

- `_prelude.sh` uses `SLURM_SUBMIT_DIR`, since Slurm copies the batch script
  to a spool directory and `$(dirname "$0")` no longer finds the folder.
- `sitecustomize.py` shims `numpy.bitwise_count` (numpy >= 2.0) because the
  site env ships numpy 1.26.  It loads automatically via PYTHONPATH.
- Smoke run 3078180 (GH200, torch 2.11.0+cu130, triton 3.5.1): all tests
  passed, Triton kernels active, sweep in `results/smoke_bf16.csv`.

## The NLP experiment (byte-level LM on enwik8)

`train_lm.py` trains a 4-layer, 4-head, d_model 256 pre-norm decoder (4.3M
params) at T=4096 on enwik8 with the attention op swapped between causal softmax
(SDPA) and SMAT with a fixed mask at d=1,2,3 (d=1 is plain causal linear
attention; the elu+1 feature map is the repo's `make_phi`).  Everything else is
identical across arms.  Reports validation bits per character overall and by
position bucket, per-byte perplexity, training-loss stats, gradient health,
ms/step and peak memory.  Data: `data/enwik8` (90M/5M/5M split).

    run_lm_interactive.sbatch        4 arms sequentially, 3000 steps      -> results/lm_T4096_seq.csv
    run_lm_20k.sbatch                4 arms, 20k steps, lr 3e-4           -> results/lm_T4096_20k.csv
    run_lm_20k_qknorm.sbatch         + QK layer-norm in every arm         -> results/lm_T4096_20k_qkn.csv
    run_lm_20k_qknorm_learnphi.sbatch + learnable phi projection          -> results/lm_T4096_20k_qkn_lphi.csv

Findings (one seed each, so differences < ~0.05 BPC are noise):

- Without QK-norm, lr 3e-4: softmax 2.00, d=1 2.46, d=3 2.32; d=2 went NaN at
  step 9250 (the elu+1 normaliser can underflow to zero; softmax has no such
  division).  Softmax also showed gradient spikes to ~150 at lr 1e-3.
- With QK-norm (numerically clean, matched across arms): softmax 1.53,
  d=1 2.98, d=2 2.44, d=3 2.53.  The mask buys ~0.5 BPC over plain linear
  attention; softmax remains far ahead at this scale.  QK-norm helps softmax
  and hurts linear attention, so the softmax-SMAT gap in that table is partly
  the stabiliser.
- Learnable phi (with QK-norm) went NaN at step 4000 for d=1: the learned
  projection can drive elu+1 features to exactly zero.  Not recommended.
- The SMAT arms train on the torch path (fp32), so ms/step and peak MB are not
  a fair efficiency comparison against bf16 flash SDPA; see below.

## Backward passes for the Triton kernels

`smat_triton_bwd.py` gives the repo's three forward-only kernels fused backward
passes as `torch.autograd.Function`s, and `patch()` swaps them into
`smat_attn`'s kernel table so `smat_attention(backend="auto")` becomes
differentiable with no change to the repo.

- `chunk_local`: dSin is one GEMM; dPhi by a row-parallel kernel, dPsi and dVb
  by a column-parallel kernel, both keeping the C x C tiles in registers.
- `pool`: one kernel; each program loads its profile's dF tile once and writes
  dPsi, dVb for the profile's columns (no atomics, columns are disjoint).
- `incidence`: C^T applied by the forward kernel with the transposed
  point-to-planes table (uniform degree, checked).

`test_triton_bwd.py` checks every gradient against the torch path (rel err
~1e-3, TF32) at T up to 8192 and d up to 4, and times fwd+bwd at the LM shapes:
1.1-1.2x at d=1,2 and 2.0x at d=3 with 45% of the memory.  `train_lm.py
--triton-bwd` trains through the kernels: identical loss trajectory, 91 -> 61
ms/step on a shared GPU, peak memory 4.1 -> 2.9 GB at d=3.

Handy: `srun --jobid=<running job> --overlap -n1 --gpus=1 ...` runs a test on
the GPU of a job that is already running, which is how these were developed
while the sweeps ran.

### T=8192 (QK-norm, SMAT arms through the fused backward kernels)

`run_lm_8k_tri.sbatch` -> `results/lm_T8192_20k_qkn_tri.csv`.  softmax 1.50,
d=1 3.18, d=2 2.87, d=3 2.98 (20k steps, one seed, no non-finite steps).  Both
linear arms are worse than at T=4096 and the mask's gain over d=1 shrank from
0.54 to 0.31 BPC; d=3 again trails d=2.  The likely reading is that the LM gain
is sparsity (fewer landmark edges per recent row; d=2 has the fewest) rather
than routing capacity, which is positional and not something text rewards.

### Control: no long-range block

`run_lm_nolr.sbatch` (`train_lm.py --no-lr`) keeps the d=2 landmark/recent
split but drops the landmark block entirely, so recent rows see only the recent
segment.  If it matches d=2, the LM gain comes from cutting recent rows off
from far context and the hyperplane geometry contributes nothing on enwik8.
Results -> `results/lm_T{4096,8192}_20k_qkn_nolr.csv`.

Result of the control (T=4096): 2.20 BPC, vs 2.44 for the full d=2 mask and
2.98 for d=1.  The landmark block *costs* 0.24 BPC on enwik8; the whole gain
over d=1 was cutting recent rows off from far context.  Mechanism: a positive
linear kernel averages every allowed key, so ~44 positionally chosen landmark
tokens dilute the local signal.

### Gate 1 (`train_lm.py --gate`)

An input-dependent per-head key gate on the landmark write only
(`h_j = sigmoid(W_h x_j)` multiplying psi_j before pooling; the causal scan and
the mask are untouched, exactness preserved).  `run_lm_gate.sbatch`.  Result:
2.209 BPC with 99.6% of gates below 0.1 -- the gate learns to switch the block
off and converges to the no-long-range control.  Gates saturate closed within
~4k steps because every landmark's average effect on its readers is dilution,
and a saturated sigmoid cannot reopen.

### Synthetic tasks

`mqar_lm.py` (`run_mqar.sbatch`): Multi-Query Associative Recall in Zoology's
layout through the same LM stack -- content-addressed, so a limits experiment
for SMAT (recent rows see ~1/47 of landmark pairs at d=2).  `needle_lm.py`:
a positional needle task on the mask's marked columns; not yet run, the marked
set needs landmark-only columns with recent-row isolation (todo).

MQAR configuration notes (2026-09-03): a 4-layer transformer with learned
absolute positions and no local mixing never found the induction circuit
(stuck at 1/n_pairs); a causal depthwise short conv (`--short-conv 3`, in every
arm) fixes that at T=256.  At T>=1024 softmax still did not converge within
~4k steps regardless of QK-norm, bf16, or pad-vs-noise filler: the recall
gradient starts at ~1/T.  Softmax solves 16 pairs at T=512 in ~1000 steps and
fails at 32, so the comparison is run at T=512, 16 pairs, 128 keys
(`run_mqar.sbatch` -> `results/mqar_T512.csv`), where d_max=3.

MQAR result (T=512, 16 pairs, 128 keys, 2500 steps): softmax 100% (from step
1500), SMAT d=1 1.3%, d=2 1.1%, d=3 1.5% -- all three linear arms at chance
(1/128).  Content-addressed recall is outside what a positive linear kernel
can do at this state size, and the mask cannot help because the needed pair is
positionally arbitrary.  This is the limits number for the paper.

### Positional needle-in-a-haystack (`needle_lm.py`, `run_needle.sbatch`)

k payload symbols planted at k landmark columns with distinct profiles; a
request token at a recent row whose mask pattern isolates needle m; target is
needle m's symbol.  enwik8 text as the haystack, identical sequences for every
arm.  T=4096, 3000 steps, one seed.  d_ref=2, k=2: softmax 100%, SMAT d=1 4%
(chance 6%), SMAT d=2 61% and still rising when the schedule ended.  The two
SMAT arms differ only in the mask: the hyperplane routing gets the needle
through real text once the model learns to suppress filler values, which takes
~2000 steps (the dilution problem of the LM runs, overcome here because the
signal is positional).  `run_needle_long.sbatch` reruns the SMAT arms at 8000
steps; `run_needle_pad.sbatch` is the constant-haystack variant.
d_ref=3, k=3 (same setup): softmax 100%, SMAT d=1 5%, SMAT d=3 98.8% at step
500 and 100% from step 1000.  The two SMAT arms differ only in the mask.

Taylor (Based) feature map test (`train_lm.py --phi taylor --taylor-dim 16`,
`run_mqar_taylor.sbatch`): SMAT d=1 on the same MQAR setting scored 1.7% at
2500 steps (elu+1: 1.3%, softmax: 100%).  Swapping the feature map alone did
not unlock content recall in this budget; the loss drifted 4.80 -> 4.74 only.
Remaining candidates: state pollution by the ~480 noise tokens (needs decay or
value suppression), learning rate (Based sweeps 1e-4..1e-2), and training
length (linear attention on MQAR is known to converge late).

### State-pollution probe (`run_mqar_probe.sbatch`, `results/mqar_T512_probe.csv`)

Measurement: share of each query row's normaliser phi(q).sum_j phi(k_j) that
comes from noise tokens, per layer.  Intervention: `--oracle-keep` zeroes every
noise token's key (the repo's key gate h with an oracle mask), so the state
holds only the kv pairs and queries.  MQAR T=512, 16 pairs, SMAT d=1:

    feature map   oracle   acc      noise share of normaliser after training
    elu+1         no       1.4%     0.84 0.81 0.88 0.89   (init ~0.89)
    elu+1         yes      100%     0 (by construction), 96.5% already at step 500
    taylor        no       1.0%     0.79 0.89 0.87 0.89
    taylor        yes      100%     99.8% at step 500

The linear kernel retrieves 16 pairs perfectly once noise is out of the state;
training never learns to push the noise keys down (share stays ~0.85).  The
blocker for content recall is the absence of a forgetting / selection
mechanism, not the feature map.  Mamba-2's input-dependent decay is exactly
that mechanism.

### Selective decay in the causal scan (`train_lm.decayed_segmented_scan`, `--decay`)

`S_t = a_t S_{t-1} + psi_t vb_t^T` with one reset at n; chunked torch
implementation, verified against the repo scan (a=1) and a brute-force
recurrence.  Landmark pooling and the mask are untouched, so masked-out
positions stay exactly non-influential; Theorem 3.2's kernel becomes
phi(q).phi(k) times the decay product along the path.  Variants:
`on` (GLA-style a_t = sigmoid(.)^(1/16), pure forget), `coupled` (write scaled
by 1-a_t), `mamba2` (dt = softplus(W x + b), A = exp(A_log) per head,
a = exp(-dt A), write scaled by dt; Mamba-2 init ranges).

MQAR T=512, 16 pairs, 2500 steps (`results/mqar_T512_decay.csv`,
`mqar_T512_mamba2.csv`): d=1 pure forget 1.2%, d=1 mamba2 2.4%, d=2 mamba2
1.6%.  Not solved in this budget.  Two engineering notes: the C x C weight
tile must be masked *before* the exp (positive differences overflow), and the
normaliser needs a floor once dt can underflow.  Likely reasons it did not
learn: the gate is gradient-starved at Mamba-2's init (softplus' ~ dt ~ 0.01),
weight decay was applied to the gate bias and A_log, and Mamba on MQAR is
known to need many more steps than softmax; untested.

Gate sweep (`run_mqar_gate_sweep.sbatch`, `results/mqar_T512_gate_sweep.csv`),
d=1, mamba2 decay, MQAR T=512 16 pairs:
    A  Mamba-2 default dt init [1e-3, 1e-1], no wd on gate params, 5k steps   4.6%
    B  dt init [0.1, 1.0], no wd on gate params, 5k steps                     100% (62% at 2k, 98% at 3k)
The gate was gradient-starved at the default init (softplus' ~ dt ~ 0.01); in
a responsive range it learns to select within ~2k steps.  SMAT + selective
decay does content-addressed recall with the elu+1 kernel unchanged.
    C  B + gate lr x5, 5k steps                                                4.9%  (worse)
    D  C at 10k steps                                                          cancelled

### Zoology MQAR (`mqar_lm.py --zoology`, `run_mqar_zoology.sbatch`)

Zoology's `multiquery_ar` generator reproduced (vocab 8192 split key/value,
every key queried once, power-law gaps with power_a=0.01, random tokens
elsewhere, label at the position after the query).  Their four standard
settings (T, pairs) = (64,4), (128,8), (256,16), (512,64); 2-layer d_model 256
models; causal short conv in every arm; SMAT arms with Mamba-2 decay
(dt init [0.1,1], no wd on gate params); chunk = T/4 so d=3 is feasible from
T=128.  8000 steps of batch 64, lr 1e-3, no lr sweep.  Results ->
`results/mqar_zoology.csv`.

First Zoology pass (`results/mqar_zoology.csv`, dt init [0.1,1], A init [1,16]):
softmax 100% everywhere; d=1 100% at T=64,128 but 0.0% at T=256 (loss pinned at
ln 4096); d=2 89%/73% and d=3 72% at T=64/128 against predicted visibility
ceilings 0.81/0.78/0.80.  Cause of the T=256 failure: with A up to 16 the
initial per-step decay exp(-dt A) is ~0 for most heads (state wiped within a
token, no gradient to recover) -- a dead init, the mirror image of the earlier
gradient-starved one; T=64/128 succeeded on lucky draws.  Rerun (`_v2`) with
dt in [0.05,0.5] and A in [1,2] so initial decays lie in [0.37,0.95].

### Global channel + mask bank on standard MQAR (`run_mqar_table.sbatch`)

Zoology generator, T=256, pairs 4..64, vocab 8192; 2-layer models at d_model
16/32/64 (head dim 16, r=64), early stop at 99%, seeds 0..2 outermost.
Arms: gated d=1 (Mamba-2-like reference); gated d=2 and d=3 with
`--n-P -1` (global channel = the kv context; the builder needs q^(d-1)
profiled columns so at 64 pairs it is capped at n_P=115 for d=2 and 79 for
d=3) and `--mask-bank` (row-type offset (layer*H + head) mod B per head, so
each head reads a different pooled state for the same row).  The decay path
now includes the Remark 3.4 global term; the bank is a roll of U before the
type-major readout.  Results -> `results/mqar_table_T256.csv`, with
state_per_head and params per row for the memory-recall comparison.

Status 2026-09-08 (stopped by the user mid-run; `results/mqar_table_T256.csv`,
`results/mqar_d1_iter.csv`): at d_model 16 every arm is dead (d=1 3.9% / 0.1%
at 4 / 8 pairs; d=2+P 6.8% / 0.3%; d=3+P 12.5%), and lr in {3e-3, 1e-2},
QK-norm off, a more responsive gate init, and r=16 did not help.  At d_model
32, gated d=1 reached only 1-2% at 4000 steps (4 pairs).  So at the widths of
the Log-Linear table our gated linear attention is far below Mamba-2's row
(47 / 75 / 90): the positive elu+1 kernel with the normaliser is a weaker
recall mechanism than SSD's signed B, C at small width, independent of the
mask.  Width 64 not reached.  Next direction chosen by the user: mask bank
(per-layer/head row-type offsets) rather than the global channel, since it
assumes nothing about where the pairs sit; `run_mqar_table.sbatch` should be
rerun with arms d=1 / d=2 --mask-bank / d=3 --mask-bank, widths in the order
64, 32, 16, and ideally more, smaller heads so H*L approaches B (=13 at T=256,
d=2).

### Mamba-2 mixer + SMAT G (`--arm mamba2smat`), 2026-09-08

The SMAT block was built around a normalised positive kernel; reconstructing
Mamba-2 inside it (signed r=128 scan, gate) still lost badly to the real
mixer at width 16 (11.5% vs 99.5% on Zoology's (64,4) cell).  So the causal
part is now the real `mamba_ssm` Mamba2 mixer, with its state and conv reset
at the landmark/recent boundary via `seq_idx` (verified exact: recent rows
change by 0.0 under landmark perturbation), and SMAT's landmark block G
(positive phi, pooled, incidence, type-major readout, own normaliser, optional
global channel and mask-bank offsets) projected to d_model and added to the
recent rows.  d=1 is literally Mamba-2.  Theory: G and its results unchanged;
the causal segments are Mamba-2's kernel; cost/decode unchanged.

Protocol correction: the Log-Linear paper's MQAR is 256-token sequences with
4..64 pairs (all at T=256), following Based's training recipe (batch 256,
32 epochs of 20k examples = 2500 steps, lr swept over logspace(-3,-1.5,4),
early stop at 99%, 5 seeds).  Earlier width-16 runs used lr <= 1e-2 and
batch 64; real Mamba-2 solved (64,4) at 1e-3 but not (256,4), so the high end
of the lr sweep is what the T=256 cells need.  `run_mqar_zoogrid16.sbatch` ->
`results/mqar_zoogrid_dm16.csv`: arms mamba2, mamba2smat d=2/3 + bank.

### Zoology pipeline (2026-09-08)

Per-cell training was the wrong protocol: Zoology (and hence Based and the
Log-Linear paper) trains ONE model on the mixture of five cells
((64,4)x100k, (128,8)/(256,16)/(256,32)/(256,64) x 20k each), 32 epochs,
batch 256, lr swept over logspace(-3,-1.5,4), tests per cell, early-stops at
valid accuracy 0.99.  Real Mamba-2 in my per-cell harness scored ~0% on
(256,4) at every lr while solving (64,4); through Zoology's pipeline the
mixture bootstraps the harder cells.  Setup: `~/zoology` (cloned), deps in
the user site (put on PYTHONPATH; the site module sets PYTHONNOUSERSITE=1),
`WANDB_MODE=disabled`.  `zoo_reference_configs.py` (attention, Mamba-2;
`run_zoo_reference.sbatch`) and `zoo_smat_configs.py` with
`zoo_smat_mixer.SmatMamba2` (Mamba-2 causal blocks with seq_idx reset + G,
mask per T cached, d_eff = min(d, d_max(T)); `run_zoo_smat.sbatch`).
Zoology's Mamba-2 needs headdim | 2*d_model, so headdim = min(64, 2*d_model).

Width-16 results so far (Zoology mixture protocol, mean over the five
in-distribution cells, one seed; logs `zoo-ref-3114372.out`,
`zoo-smat-overlap-d2.out`, `zoo-mamba2-overlap.out`):
    attention   lr 1e-3 92.9 | 3.16e-3 94.0 | 1e-2 84.9 | 3.16e-2 ~98 (99/99/98/96/98)
    SMAT d=2+bank (Mamba-2 causal blocks)  lr 1e-3 22.9 | 3.16e-3 22.6 | 1e-2 25.2 | 3.16e-2 unstable
                per cell at best lr: 72 / 53 / 0.6 / 0.2 / 0.1  -- the 16..64-pair cells
                route every query through G; with 1 head x 2 layers the bank covers
                2 of 7 hyperplanes.  Even 4/8 pairs sit below the visibility ceiling.
    Mamba-2 (real mixer, d_state 128)  lr 1e-3: 72% mean at epoch 9 and rising
                (93/89/79/62/35), above the Log-Linear table's 46.9 -- their Mamba-2
                config (d_state?) is not given; ours is the reference that counts.
Conclusion at this width: the SMAT gap is the mask (reset + 1/q visibility x bank
coverage), with the identical mixer inside.  Remaining: Mamba-2 full sweep, SMAT
d=3 and d=1 (`run_zoo_rest.sbatch`, job 3114752); then widths 32/64, more heads,
global channel, d_state-matched rows.

NOTE 2026-09-08 17:00: a second session is editing this folder concurrently
(`run_zoo_pair.sbatch`, ZOO_LRS support in `zoo_smat_configs.py`).  Check
mtimes before assuming a file's state.  Mixer now: Zoology's vendored Mamba2
(its slow path has no seq_idx; reset is done by folding the two equal segments
into the batch), G output zero-initialised.  Earlier SMAT rows through the
mamba_ssm-based wrapper are INVALID: that wrapper's d=1 control did not learn
(0.8% after 8 epochs) while Zoology's Mamba-2 reached 83%.

### Multi-resolution form: Mamba-2 as a point in SMAT's parameter space (2026-09-08)

`zoo_smat_mixer.SmatMamba2Shared` (config: `ZOO_SHARED=1`).  Analogous to how
log-linear attention reduces to Mamba-2: G becomes an extra resolution level
of the same masked-kernel expression rather than a bolted-on branch.
  level 0  : the un-reset SSD state (Mamba-2's own recurrence, its decay)
  level d  : SMAT's hyperplane block G, keys phi(B_t), queries phi(C_t), values x_t
             taken from Mamba-2's own in_proj/conv stream (positive phi keeps G's theory)
  y_t     += lambda_t * G_t,  lambda_t = alpha_h * sigmoid(w . u_t),  alpha init 0
G is added before Mamba-2's gated RMSNorm and out_proj (shared).  Verified on
GPU: alpha=0 output equals stock fused Mamba-2 bit-for-bit (max diff 0.0);
alpha>0 changes recent rows only; gradients reach alpha.  Optional hard reset
via seq_idx (`reset=True`) for the ablation; by default the boundary decay is
Mamba-2's own learned a_t.  Theory: G's exactness / non-influence / VC=d hold
for its contribution; layer-level exactness is conditional on lambda^(0)=0.
Mamba-2's non-fused path needs xBC contiguous for causal_conv1d (fixed).

Ablation table at width 16 (Zoology protocol) so far: attention ~98 |
Mamba-2 83-85 | SMAT d=2 with reset 25 | parallel hybrid (separate q/k/v,
random-init out proj, no lambda) 33.5 @1e-3, ~70 partial @3.16e-3 -- below
Mamba-2 at matched epochs (random-init G injects noise, delays the
transition).  Running: plain hybrid remaining lrs, with-reset d=3/d=1,
multi-resolution d=2 (`logs/zoo-shared-d2.out`).

### Width-16 Zoology table, status 2026-09-08 18:25 (compiled from logs/; mean = per-cell mean over the 5 in-distribution cells)

| arm (dm16, 2 layers) | lr | epochs | mean | 4 / 8 / 16 / 32 / 64 pairs | log |
|---|---|---|---|---|---|
| attention (Zoology ref) | 3.16e-2 | 32 | 78.2 | 100 / 100 / 99 / 99 / 98 | zoo-ref-3114372 |
| Mamba-2 (Zoology ref, d_state 128) | 1e-3 | 32 | 83.2 | 99 / 98 / 94 / 80 / 44 | zoo-rest-3114752 |
| Mamba-2 (Zoology ref) | 3.16e-3 | 14 (cut) | 85.1 | 99 / 97 / 93 / 82 / 55 | zoo-rest-3114752 |
| SMAT d=1 wrapper control (== Mamba-2 in our wrapper) | 3.16e-3 | 32 | 72.4 | 95 / 88 / 78 / 62 / 39 | zoo-wrapper-d1 |
| SMAT d=2 hybrid (no reset, G added) | 3.16e-3 | 32 | 72.5 | 95 / 90 / 79 / 61 / 38 | zoo-hybrid-d2-b |
| SMAT d=2 hybrid | 1e-2 | 31 | 79.8 | 95 / 92 / 85 / 73 / 53 | zoo-hybrid-d2-b |
| SMAT d=2 hybrid | 3.16e-2 | running | 68.8 @ep1 | 91 / 85 / 76 / 58 / 34 | zoo-hybrid-d2-b |
| SMAT d=3 with reset | 1e-3 | 5 (killed) | 0.9 | 3 / 1 / 0 / 0 / 0 | zoo-reset-d3d1 |
| SMAT d=2 shared-hybrid (SmatMamba2Shared) | 3.16e-3 | 32 | 45.0 | 94 / 76 / 40 / 12 / 3 | zoo-shared-d2-v2 |
| SMAT d=2 shared-hybrid | 1e-3 | 32 | 39.7 | 79 / 59 / 38 / 17 / 6 | zoo-shared-d2-v2 |

Reading: (1) the d=1 control matches the d=2 hybrid at the same lr (72.4 vs 72.5), so at width 16 G neither helps nor
hurts recall once the recurrence is unreset; (2) both sit ~13 points below Zoology's own Mamba-2 at the same lr
(85.1 at epoch 14), so the remaining gap is the wrapper (TransformerBlock + Identity state mixer vs the reference
Mamba2Block and its init), not the mask -- this is what `SmatMamba2Block` in `zoo_smat_mixer.py` (other session,
18:21) targets; (3) with the boundary reset, d=3 is at chance after 5 epochs: the reset removes the recurrence's
recall and G at 1/q visibility cannot replace it at this width; (4) the first shared-hybrid variant underperforms
the separate-projection hybrid. Batch job 3114926 (run_zoo_rest2) was cancelled at 0 s by the concurrent session.


**Fix (job 3115485, `run_zoo_m2blk.sbatch`, 18:56):** `SmatMamba2Block` in `zoo_smat_mixer.py` hosts the SMAT mixers
inside Zoology's own `Mamba2Block` (RMSNorm, no MLP, add->norm->mixer, and Zoology's Mamba init path, which leaves
the mixer's kaiming init alone instead of re-initialising every Linear to std 0.02). `zoo_smat_configs.py` installs
it via `zoology.mixers.mamba2.Mamba2Block = SmatMamba2Block` and uses `block_type="Mamba2Block"`; run names carry
`_m2blk`. Same data, protocol and lrs as before.

| arm (dm16, 2 layers, Mamba2Block) | lr | epochs | mean | 4 / 8 / 16 / 32 / 64 pairs | log |
|---|---|---|---|---|---|
| SMAT d=1 (== Mamba-2; wrapper check) | 3.16e-3 | 32 | 85.0 | 99 / 98 / 95 / 83 / 50 | zoo-m2blk-d1 |
| SMAT d=1 (== Mamba-2) | 1e-2 | 32 | 89.6 | 100 / 99 / 98 / 90 / 61 | zoo-m2blk-d1 |
| SMAT d=2 hybrid (Mamba-2 + G, no reset, own q/k/v) | 3.16e-3 | 32 | 87.0 | 99 / 99 / 96 / 86 / 55 | zoo-m2blk-hybrid-d2 |
| SMAT d=2 hybrid | 1e-2 | 32 | **90.2** | 100 / 100 / 98 / 91 / 63 | zoo-m2blk-hybrid-d2 |
| SMAT d=2 multi-resolution (SmatMamba2Shared) | 3.16e-3 | 32 | 84.4 (peak 84.8) | 99 / 97 / 94 / 82 / 51 | zoo-m2blk-shared-d2 |
| SMAT d=2 multi-resolution | 1e-2 | 32 | 87.8 | 99 / 99 / 97 / 88 / 57 | zoo-m2blk-shared-d2 |

Job 3115485 finished (all 6 runs, 32 epochs each, single seed). Epoch-matched at 3.16e-3 (Mamba-2 ref / d=1 wrapper / hybrid d=2 / multi-res d=2): ep1 0.6 / 0.5 / 58.6 / 64.9;
ep2 62.6 / 61.7 / 77.5 / 75.4; ep3 73.8 / 72.2 / 81.4 / 78.9; ep4 79.2 / 78.1 / 83.9 / 80.4; ep8 84.1 / 83.8 / -- / 83.7.
Reading: the d=1 wrapper now reproduces the reference (84.8 vs 85.1), so the earlier 13-point gap was entirely the
block/init, and every `TransformerBlock` SMAT row above is superseded. With the block fixed, both d=2 forms train
*faster* than Mamba-2 (the G path skips Mamba-2's epoch-1 plateau: 59-65% at epoch 1 vs 0.6%) and the plain hybrid ends
above Mamba-2 at both lrs (+2.0 at 3.16e-3, +0.6 at 1e-2, gain concentrated in the 32/64-pair cells); the
multi-resolution form ends slightly *below* Mamba-2 at both lrs (-0.6 / -1.8), so sharing B/C/x with the SSM costs
more than the extra resolution buys at this width. Margins are 1-2 points on one seed: seeds and widths 32/64 next.
Comparability with Log-Linear Attention's Table 2 (checked 2026-09-08 against arxiv 2506.04761v3 and the
HanGuo97/log-linear-attention repo): the column is labelled only "Dimension" (16/32/64); the appendix says the setup
follows Arora et al. (Zoology), whose swept axis is model dimension, so dm16/32/64 is the matching reading.  Their
Mamba-2 state size / head layout for that table is not stated anywhere and the repo has no MQAR config; ours is
Zoology's default (d_state 128).  Their Mamba-2 at "16" is 46.9 (5 seeds) vs our reference 85-90, so a direct
paste into their table is not defensible without knowing their config.

### Multi-resolution rev 3 (`SmatMamba2MR`, `ZOO_MR3=1`; job 3116675, `run_zoo_m2blk.sbatch` -> `run_zoo_mr3.sbatch`), width 16, Mamba2Block

Fixes aimed at the Log-Linear framing (G as an extra level of the SAME recurrence, sharing B/C/dt.x, Mamba-2 at lambda=0):
head-count knob so heads x layers >= B (mask offsets per head), G values = Mamba-2's write-gated dt_t x_t, identity kernel
(keys B_t, queries C_t, no feature map / normaliser), lambda_t = act(alpha[h, rho(t)] + w_h . u_t).  Overhead over Mamba-2:
648 params/layer (lam_w + alpha table) vs 1,088 for the plain hybrid's own q/k/v/out.  Smoke: lambda->0 equals Mamba-2 to 7e-6.

| arm (dm16, 2 layers) | lr | epochs | mean | 4 / 8 / 16 / 32 / 64 | log |
|---|---|---|---|---|---|
| Mamba-2 control, headdim 4 (8 heads) | 1e-2 | 32 | 89.0 | 100 / 100 / 98 / 89 / 59 | zoo-mr3-d1-hd4 |
| Mamba-2 control, headdim 4 | 3.16e-3 | 32 | 85.0 | 99 / 99 / 95 / 83 / 50 | zoo-mr3-d1-hd4 |
| MR3 d=2 headdim 8, softplus lambda (unbounded) | 1e-2 | 32 | 88.0 | 99 / 100 / 98 / 88 / 56 | zoo-mr3-d2-variants |
| **MR3 d=2 headdim 8, sigmoid lambda (bounded)** | 1e-2 | 32 | **90.3** | 100 / 100 / 99 / 91 / 62 | zoo-mr3-d2-hd8-sig |
| MR3 d=2 headdim 4, softplus | 1e-2 | 32 | 76.8 (unstable: 4-pair cell collapsed from ep3) | 94 / 90 / 84 / 70 / 46 | zoo-mr3-d2-hd4 |
| MR3 d=2 headdim 4, softplus | 3.16e-3 | 11 (cut) | 87.2 (ahead of control at every epoch) | 99 / 98 / 95 / 86 / 59 | zoo-mr3-d2-hd4 |
| MR3 d=2 headdim 4 + Mamba-2 decay in G | 1e-2 | 7 | diverged to chance at ep7 (killed) | | zoo-mr3-d2-decay-fix2 |

Epoch-matched, lr 1e-2 (control / hd8-softplus / hd8-sigmoid): ep8 84.4 / 85.8 / 86.2; ep14 85.4 / 86.1 / 87.7; ep18 86.6 / 86.8 / 88.7;
ep32 89.0 / 88.0 / 90.3.  Reading: (1) the bounded gate is what makes the multi-res form work -- with unbounded lambda the
8-head arms blow up on short sequences at lr 1e-2 and the 4-head arm ends level with the control; with sigmoid lambda the
same 4-head arm leads the control at every epoch from 8 on and ends +1.3 (gain in the 32/64-pair cells: +2 / +3);
(2) the headdim-4 collapse was step size, not the mask: at 3.16e-3 it leads the control (87.2 @ ep11 vs 85.0 final);
(3) decay inside G is harmful (no reason to decay landmark states that the query wants undecayed);
(4) caveat: the winning arm is headdim 8 (4 heads) while the control is headdim 4 (8 heads); Mamba-2 is insensitive to
the split (89.6 @ 1 head, 89.0 @ 8 heads) but a headdim-8 control belongs in the seed sweep.  Single seed throughout.
Configuration adopted for SMAT going forward: `ZOO_MR3=1 ZOO_HEADDIM=8 ZOO_LAMACT=sigmoid`.

**Seed sweep (head-matched: Mamba-2 headdim 8 vs MR3 bounded-lambda headdim 8, lr 1e-2, seeds 1/2/3), launched as
overlap steps on job 3116675 -- CUT at 23:50 when the job's own batch script finished (the parent job ending kills
overlap steps; the sbatch needs a trailing hold if it hosts overlap work).  Epoch-matched partial results:**

| seed | epoch | Mamba-2 hd8 | MR3 hd8 sigmoid | log |
|---|---|---|---|---|
| 1 | 11 | 77.7 | 77.3 (diverged to chance at ep8, recovered) | zoo-seed-{ctrl,mr3}-hd8-s1 |
| 2 | 15 | ~82 | 77.4 | zoo-seed-{ctrl,mr3}-hd8-s2 |
| 3 | 11 | 85.7 | 85.8 | zoo-seed-{ctrl,mr3}-hd8-s3 |
| ctrl only | 24 | s2 85.0, s3 86.7 | | |

Reading: with matched heads and fixed config, MR3 does NOT lead Mamba-2 (one seed level, one behind, one unstable);
seed-to-seed spread of Mamba-2 itself is ~9 points at epoch 4 and ~8 at epoch 24.  The single-seed +1.3 above was
selection among 5 variants on one seed.  Conclusion at d_state 128: the multi-resolution form matches Mamba-2 and
adds nothing measurable on MQAR.  Next: the small-state regime (d_state 16, job 3116795, `run_zoo_ds16.sbatch`),
where Mamba-2's single state saturates and extra profile states can matter.

### Small-state sweep (headdim 16, d_state 16; job 3116795, `run_zoo_ds16.sbatch`), lr 1e-2, Mamba2Block, single seed, 2026-09-09

Cancelled by the user at ~29/32 epochs (curves flat).  Mamba-2 control = the same `SmatMamba2MR` class with d=1.

| d_model (heads) | Mamba-2 hd16/ds16 | SMAT multi-res bounded-lambda | Log-Linear Table 2: Mamba-2 / w/ Log-Linear |
|---|---|---|---|
| 16 (2) | **46.9** (91/74/49/17/3), 32 ep | 40.8 (83/65/39/14/2), 31 ep | 46.9 / 55.9 |
| 32 (4) | **82.8** (100/99/96/80/40), 32 ep | 71.7 (98/96/88/57/20), 29 ep | 75.1 / 76.5 |
| 64 (8) | 81.5 (100/99/97/77/35), 29 ep | 82.4 (100/99/97/78/37), 23 ep | 89.6 / 92.9 |

Reading: (1) our Mamba-2 at width 16 reproduces Log-Linear's 46.9 exactly, so their "Dimension" column sets headdim and
d_state together with d_model (resolves the comparability question above; their width-32/64 Mamba-2 rows differ from
ours by lr/epochs); (2) in the saturated-state regime the multi-resolution form is BEHIND Mamba-2 by 6 (w16) and 11 (w32)
points and level at w64, whereas Log-Linear gains +9 / +1.4 / +3.3 on the same baseline.  The extra profile states do
not add usable capacity: G's profiles are position-interleaved, ungated sums of 16-dim keys, so a query reading a
profile gets a superposition of unrelated pairs; Log-Linear's buckets are contiguous and gated.  Together with the
cut seed sweep (no lead at d_state 128), the multi-resolution SMAT form has no MQAR case at any state size tried.
Remaining positive MQAR result: the plain hybrid (own q/k/v) +0.6..2 on one seed at d_state 128, at ~+30% params at scale.

### Option 1: per-profile delta-rule states (`ZOO_POOL=delta`, job 3116975, `run_zoo_ds16_delta.sbatch`), hd16/ds16, lr 1e-2 -- cancelled by user at epoch 12-16

Each profile's state built by the delta rule on L2-normalised keys (S += beta k (v - k^T S)^T) instead of the additive
sum; positional assignment / incidence / bank unchanged; +1 linear (d_model x heads) for beta.  Verified vs brute force
(2e-7); lambda->0 == Mamba-2 (1e-6).

| width | Mamba-2 | sum-pool MR3 | delta MR3 (epoch-matched to sum-pool) |
|---|---|---|---|
| 16 | 47 | 41 | 43 @ ep16 (sum-pool 40 @ ep16) |
| 32 | 83 | 72 | 66 @ ep15 (sum-pool 67 @ ep15) |
| 64 | 82 | 82 | 78 @ ep12 (sum-pool 80 @ ep12; spiked 86 @ ep3 then diverged to 0 @ ep4, recovered) |

Reading: delta == sum-pool within 1-2 points at every width.  Interference inside a profile is NOT the bottleneck;
the positional read is (a query reaches only the profile its row type points to).  Option 2 (content read over all
profiles, `ZOO_READ=content`, on top of delta) is job 3116992.

### Option 1 + 2: delta-rule profile states + content-addressed read over all profiles (`ZOO_POOL=delta ZOO_READ=content`), hd16 / ds16, lr 1e-2, 32 epochs, single seed, 2026-09-09 (job 3116992, `run_zoo_ds16_cread.sbatch` + overlap `logs/cread_d3.sh`)

The read no longer goes through the incidence table: each profile x gets a summary key kappa_x = sum of its unit keys;
a query C_t scores all N0 summaries, softmax (learned per-head temperature), and reads sum_x w_x (C_t^T S_x).  Profile
states S_x are per-profile delta-rule memories (option 1).  lambda-gated as before; Mamba-2 exactly at lambda = 0.
Extra params over Mamba-2: lam_w, alpha table, beta_w, tau (< 0.2% of the model).  Decode Theta(N0 rp) per token,
prefill Theta(T N0 rp) = Theta(T^{2-1/d} rp); memory N0 rp per head with N0 = T^{1-1/d}.
The incidence / mask-bank offsets are unused on the read path, so the paper's exact-non-influence and positional
routing-capacity results do not describe this variant.

| d_model (heads) | Mamba-2 hd16/ds16 | Log-Linear Table 2 (Mamba-2 -> w/ LL) | SMAT d=2 content read | SMAT d=3 content read |
|---|---|---|---|---|
| 16 (2) | 46.9 (91/74/49/17/3) | 46.9 -> 55.9 | 72.8 (90/80/85/69/39) | **80.3** (87/73/77/87/78) |
| 32 (4) | 82.8 (100/99/96/80/40) | 75.1 -> 76.5 | 86.5 (98/91/93/85/56) | **93.7** (94/76/90/95/87) |
| 64 (8) | 81.5 (100/99/97/77/35) | 89.6 -> 92.9 | **98.5** (100/100/100/100/94) | 94.1 (99/98/88/94/91) |

(N0 at T=256: 13 profiles x 10 tokens at d=2; 49 x 3 at d=3.  T=64 cells fall back to d=2 for both arms.)
Reading: (1) every cell beats Mamba-2 and Log-Linear's reported gain on the same baseline; the gain is in the 32/64-pair
cells (w16: 17/3 -> 87/78); (2) d=3 wins at 2-4 heads (sharper profile labels: 3 keys vs 10 in 16 dims), d=2 wins at
8 heads (98.5 = full-attention level) with 3.8x less profile memory -- d is a memory/selection knob, not a fixed choice;
(3) the short cells (T=64/128) regress for the first ~15 epochs and recover late as lambda learns to stand down on short
inputs; a length-aware gate would likely remove the dip; (4) delta states alone (option 1) gave nothing -- the content
read is what unlocks the profile states; (5) caveats: single seed, lr 1e-2 only, and the SMAT arms hold 14x (d=2) /
50x (d=3) Mamba-2's recurrent memory at T=256 (same kind of comparison as Log-Linear's log T states; Mamba-2 at
d_state 128, i.e. 8x memory, reached 89.6 at width 16 in the earlier sweep -- a state-matched column belongs in the
final table).  Next: 3 seeds of both arms, state-matched Mamba-2 (d_state 128/512 at hd16), lr 3.16e-3, then a GDN base.

### 5-seed table, Mamba-2 base (seed 123 + seeds 1-4; jobs 3117180 / 3117181, `run_zoo_seeds5.sbatch`), hd16 / ds16, lr 1e-2, 32 epochs, early stop at 99%, 2026-09-09

| width | Mamba-2 (ours, 5 seeds) | Log-Linear paper: Mamba-2 / w/ Log-Linear | SMAT d=2 (5 seeds) | SMAT d=3 (5 seeds) |
|---|---|---|---|---|
| 16 | 45.6 (4.7) | 46.9 (2.3) / 55.9 (9.1) | 72.1 (6.0) | **75.2 (15.3)** |
| 32 | 75.1 (6.3) | 75.1 (4.9) / 76.5 (4.8) | 86.6 (2.8) | **94.5 (2.1)** |
| 64 | 87.3 (6.1) | 89.6 (6.1) / 92.9 (2.7) | 95.9 (3.5) | **95.5 (2.0)** |

Per-seed values (seed 123, 1, 2, 3, 4):
16 Mamba-2: 46.9, 41.8, 40.6, 52.6, 45.9
16 SMAT d=2: 72.8, 75.0, 75.8, 61.7, 75.4
16 SMAT d=3: 80.3, 88.8, 82.2, 49.1, 75.5
32 Mamba-2: 82.8, 80.7, 68.4, 70.8, 72.7
32 SMAT d=2: 86.5, 86.1, 84.1, 91.3, 85.2
32 SMAT d=3: 93.7, 95.9, 91.6, 94.2, 96.9
64 Mamba-2: 81.5, 92.5, 82.7, 84.6, 95.1
64 SMAT d=2: 98.5, 98.3, 92.5, 91.6, 98.6
64 SMAT d=3: 94.1, 99.0, 94.8, 94.9, 94.8

Reading: our 5-seed Mamba-2 reproduces Log-Linear's Mamba-2 row at every width (45.6 vs 46.9, 75.1 vs 75.1, 87.3 vs 89.6).
SMAT d=3 content read: +30 / +19 / +8 over Mamba-2 and +19 / +18 / +3 over Log-Linear's reported variant, with 2-6 point
std except width 16, where seed 3 never engaged the profile memories (49.1, long cells at Mamba-2 level) -- an optimisation
failure at lr 1e-2, not capacity.  d=2 is steadier at width 16 (72.1 +- 6.0) and matches attention at width 64.
Memory caveat as before: 14x (d=2) / 50x (d=3) Mamba-2's recurrent memory at T=256 vs Log-Linear's ~9x.

### 5-seed table, Gated DeltaNet base (`SmatGDN`, `ZOO_BASE=gdn`; jobs 3117325 / 3117381 / 3117441, `run_zoo_gdn_seeds.sbatch`), fla 0.5.2 GatedDeltaNet head_dim 16 / expand_v 1 (16x16 state per head, 2*d_model/16 heads), lr 1e-2, 32 epochs, early stop at 99%, 2026-09-09

Same method on GDN: profile memories built with GDN's own L2-normalised keys, values and beta (no extra write gate),
content read over all profiles, lambda-gated before GDN's gated norm; exactly fla's GDN at lambda = 0 (max diff 0.0).
Extra params: lam_w + alpha + tau (164 / layer at width 16).  Needs triton 3.7.1 (`~/triton371`) + `FLA_TILELANG=0`.

| width | GDN (ours, 5 seeds) | Log-Linear paper: GDN / w/ Log-Linear | SMAT-GDN d=2 (5 seeds) | SMAT-GDN d=3 (5 seeds) |
|---|---|---|---|---|
| 16 | 64.8 (8.2) | 38.4 (1.0) / 40.0 (1.4) | 78.7 (9.7) | **87.6 (13.0)** |
| 32 | 74.7 (14.9) | 79.0 (2.1) / 84.4 (1.2) | 94.8 (2.2) | **98.6 (0.8)** |
| 64 | 89.5 (7.1) | >=99 / >=99 | 95.8 (2.9) | **98.3 (0.9)** |

Per-seed values (seed 123, 1, 2, 3, 4):
16 GDN: 72.1, 73.1, 63.7, 52.9, 62.2
16 SMAT-GDN d=2: 66.6, 80.6, 71.3, 84.5, 90.5
16 SMAT-GDN d=3: 88.1, 66.4, 87.3, 98.0, 98.4
32 GDN: 86.5, 59.0, 80.9, 88.8, 58.4
32 SMAT-GDN d=2: 92.7, 96.2, 94.8, 92.7, 97.6
32 SMAT-GDN d=3: 97.1, 99.0, 99.0, 98.6, 99.1
64 GDN: 90.5, 93.2, 92.8, 93.9, 76.9
64 SMAT-GDN d=2: 97.9, 96.7, 99.0, 92.4, 93.1
64 SMAT-GDN d=3: 99.2, 99.0, 98.5, 96.9, 98.1

Reading: our GDN control is far above Log-Linear's GDN row at widths 16/32 (64.8 vs 38.4, 74.7 vs 79.0 with a
15-point seed split), so their GDN config differs from ours; only within-table deltas are comparable.  SMAT-GDN d=3:
+23 / +24 / +9 over its base, at the attention ceiling (98-99) on every seed at widths 32 and 64 (most early-stopped
before epoch 15), vs Log-Linear's reported +1.6 / +5.4 / 0 on GDN.  Width 16 has the same one-seed gate-engagement
failure as the Mamba-2 base (seed 1: 66.4).  d=2 is within 1-4 points of d=3 at widths 32/64 with 3.8x less memory.

### Log-Linear-head-matched GDN table (`run_zoo_gdnm_seeds.sbatch`; jobs 3118352 / 3118431 / width-64 job), heads 1 / 2 / 2 as stated by Guo et al., head_dim 16, expand_v 1, lr 1e-2, 5 seeds, 2026-09-09

(expand_v 2 at one head faults intermittently in fla 0.5.2's chunk kernel on GH200 -- 9/12 launches -- so expand_v 1 everywhere.)

| width (heads) | GDN (ours, 5 seeds) | Log-Linear paper: GDN / w/ Log-Linear | SMAT-GDN d=2 | SMAT-GDN d=3 |
|---|---|---|---|---|
| 16 (1) | 45.6 (11.0) | 38.4 (1.0) / 40.0 (1.4) | 78.7 (7.5) | 76.0 (23.0) |
| 32 (2) | 67.9 (9.6) | 79.0 (2.1) / 84.4 (1.2) | 79.0 (16.4) | 82.7 (18.7) |
| 64 (2) | 68.1 (10.8) | >=99 / >=99 | 91.8 (1.6) | 86.3 (23.9) |

Per-seed values (seed 123, 1, 2, 3, 4):
16 GDN: 29.6, 41.8, 44.8, 56.8, 54.8
16 SMAT-GDN d=2: 85.2, 75.4, 68.5, 86.9, 77.7
16 SMAT-GDN d=3: 91.4, 45.2, 57.4, 95.5, 90.3
32 GDN: 75.7, 57.4, 75.9, 72.9, 57.6
32 SMAT-GDN d=2: 84.6, 79.5, 87.4, 50.9, 92.4
32 SMAT-GDN d=3: 94.9, 91.2, 53.0, 76.2, 98.4
64 GDN: 76.5, 72.7, 78.4, 57.0, 56.1
64 SMAT-GDN d=2: 91.0, 94.4, 90.5, 92.4, 90.6
64 SMAT-GDN d=3: 97.6, 95.8, 97.6, 96.8, 43.5

Reading: (1) with their head counts our GDN baseline matches theirs only at width 16 (45.6 vs 38.4); at widths 32/64
ours is 11-30 points BELOW theirs (67.9 vs 79.0, 68.1 vs >=99), so their GDN head_dim must scale with width (fla
convention) -- head_dim 16 with 2 heads is starved at width 64.  A faithful rerun needs head_dim = width, heads 1/2/2.
(2) SMAT still adds +33 / +11..15 / +24..18 mean, but with 1-2 heads the lambda gate fails to open on 4 of 30 SMAT
runs (those seeds land at or below the control); last night's 4-8-head versions had 1 failure in 30.  Gate init /
warmup fix is now required before the width-16/32 columns are quotable.

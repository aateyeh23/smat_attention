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

### Paper mask M^(d) with Mamba-2 weights and boundary reset (`ZOO_RESET_MR=1 ZOO_GDECAY=1 ZOO_LAMACT=one ZOO_READ=type`), hd16/ds16, width 16, seed 123, lr 1e-2, 2026-09-10

Exact implementation of the block mask: Mamba-2 recurrence restarted at n = T/2 (smoke: first half == Mamba-2 on the first
half, 4e-7; G off -> second half == Mamba-2 restarted, 2e-7), G = decayed pooled read through C with the mask bank, added
directly (per-head scale init 1).  d=1 reproduces Mamba-2 (42 @ ep17 vs 42; 77 vs 78 at width 32).

| arm | final | cells | note |
|---|---|---|---|
| d=1 (== Mamba-2) | 46.9 | 91/74/49/17/3 | reference |
| d=2 both layers, strict 0/1 G | 0.3 @ ep8 | chance | killed; 2 heads x 2 layers cover 4/13 row types |
| layer 0 d=1, layer 1 d=2 strict | 26 @ ep13 | 66/44/14/7/2 | stopped; worse than two Mamba-2 layers |
| d=2, G zeros -> 1/q (`ZOO_GFLOOR=q`) | 1 @ ep3 | chance | cancelled |
| d=2, G = 1+1/q on C, 1-1/q off C (`ZOO_GFLOOR=pm`) | 46.0 | 90/76/46/15/3 | == Mamba-2 |
| d=2, G = 1 on C, 1-1/q off C (`ZOO_GFLOOR=m`) | 41.7 | 82/65/40/17/4 | ~Mamba-2 (ahead early: 38 @ ep4 vs 33, then level) |

Reading: a fixed-weight read of the landmark half is worth exactly one merged table (Mamba-2's capacity) regardless of the
positional tilt; the strict sparse mask is worse than that because it hides 9/13 of the pairs.  Consistent with the
whole positional-read series.

### Content-hashed SMAT (Abdullah's `content_addr.ContentAssign`) in the Zoology pipeline, 2026-09-11 (job 3132062, `run_zoo_chash.sbatch`)

Paper mask C, pooling and decode unchanged; profile = hash(key-side input), row type = hash(query input), learned hash
annealed soft->hard over 2500 steps, balance 0.01 via Zoology's `get_auxiliary_loss`; Mamba-2 weights, boundary reset,
G added with a per-head scale (init 1).  Width 16, hd16/ds16, lr 1e-2, seed 123, 32 epochs.  `hconv` = learned depthwise
causal conv (width 4, init uniform over lags) on the key-side hash input instead of Abdullah's fixed shift of 1.
Reference rows at this setting: Mamba-2 46.9; positional d=2 with the same weights: chance; content-read bank 72.8 / 80.3.

| arm | final | best | cells 4/8/16/32/64 |
|---|---|---|---|
| d=2 point, learned conv | **86.1** | 86.5 | 88/81/88/90/84 |
| d=2 point, fixed shift 1 | 37.8 | 37.9 | 79/60/35/12/2 (hash never engaged) |
| d=3 plane, learned conv | 67.6 | 74.8 | 84/77/80/63/35 |
| d=3 plane, fixed shift 1 | 72.1 | 79.3 | 90/78/70/58/66 |

Reading: the paper's own architecture -- sparse C, pooled profiles, Theta(1) decode -- reaches 86 at the hard end
(64 pairs, 16x16 state) once the assignment is by content, +39 over Mamba-2 and above the content-read bank's d=2
(72.8).  The learned conv on the hash input is the more robust key-side source at d=2 (the fixed-shift run stayed at
Mamba-2 level: a hash-engagement failure, the same one-seed failure mode seen before).  d=3 (49 cells of ~3) is worse
than d=2 (13 cells of ~10) here, the opposite of the content-read bank: with hashing, more cells means more hash
misses (hash_hit falls), and the plane read at d=3 sums q cells.  Curves are jumpy at lr 1e-2 (d=3 sh1: 79 -> 50 -> 79).
Single seed; the 5-seed repeat and the width-32 column are next.

### Affine-subspace family under content hashing (codimension-c reads), width 16, hd16/ds16, seed 123, learned hash + key-side conv, anneal 2500, balance 0.01 (job 3132250, `run_zoo_codim.sbatch`), 2026-09-11

`ContentAssign(codim=c)`: read the codim-c flat through the query's cell (c=1 hyperplane ... c=dim point);
`_grouped_flats` enumerates all full-rank coset partitions.  The hashed path bypasses the recent-block type-cycle
cap on d (only q^{d-1} <= n binds), so d=4 (q=5, 125 cells) is buildable at T=256.

| d (q, cells at T=256) | c=1 | c=2 | c=3 |
|---|---|---|---|
| 2 (13, 13) | **86.1** (point; done earlier) | | |
| 3 (7, 49) | 67.6 final / 79.3 peak (earlier) | 72.0 final (85/72/61/66/74), collapsed to 32 mid-run | |
| 4 (5, 125) | 58 @ ep20, best 65 (killed) | **84.2 @ ep14, 83 @ ep16, still rising, no spikes (killed)** | 42 @ ep21, best 68 then collapsed to 10 (killed) |

Reading: (1) at d=4 the intermediate codimension is far ahead of both endpoints (84 vs 65 / 68), and level with
d=2 -- so performance need not fall with d if c is chosen (c ~ (d-1)/2); the family is real; (2) d=3, which has no
intermediate c, lands at 68-72 with both reads; both d=3 runs and the d=4 endpoints show anneal-time spikes /
collapses, the d=4 line read never did; (3) the d=3 point read's cells are inverted (64-pair 74-82, 16-pair 36-61):
keys are hashed correctly, filler contaminates the cells -- a hash-recipe symptom, not a geometry one.  Killed the
d=4 runs at epochs 16-21 on request.  Next (queued, job 3132379): frozen embedding hash for the two point reads, and
d=4 with heads split c=[1,2].

### Content-hashed SMAT (Abdullah's `content_addr.ContentAssign`) in the Zoology pipeline, width 16, hd16 / ds16, seed 123, 2026-09-11 (job 3132062, `run_zoo_chash.sbatch`)

Paper mask C + pooling unchanged; profile = hash(key-side input), row type = hash(query input); Mamba-2 weights
(identity kernel, dt writes, decay folded into G), recurrence reset at the boundary, G added directly (per-head scale
init 1); learned hash annealed soft->hard over 2500 steps, balance penalty 0.01 via Zoology's get_auxiliary_loss.
Key-side source: fixed shift 1 (his recipe) or a learned depthwise causal conv (width 4, init uniform over lags).
Hash adds 628 params/layer.

| arm | mean | 4 / 8 / 16 / 32 / 64 |
|---|---|---|
| Mamba-2 (d=1), same setting | 46.9 | 91/74/49/17/3 |
| positional d=2, paper mask | 0.3 (chance) | |
| content-read bank (not SMAT), d=2 / d=3 | 72.8 / 80.3 | |
| **chash d=2 point, conv** | **86.1** | 88/81/88/90/84 |
| chash d=3 plane, shift 1 | 72.1 | 90/78/70/58/66 |
| chash d=3 plane, conv | 67.6 | 84/77/80/63/35 |
| chash d=2 point, shift 1 | 37.8 (hash never engaged; long cells at Mamba-2 level) | 79/60/35/12/2 |

Reading: content addressing makes the paper's own architecture recall at the hard end (64 pairs in a 16x16 state):
d=2 point + learned conv reaches 86.1, +39 over Mamba-2 and above the content-read bank, with sublinear memory and
Theta(1) decode intact.  The learned conv on the key-side hash input is the more robust choice: the fixed shift-1
run at d=2 never engaged (one more instance of the engagement failure), while conv engaged immediately (68 at epoch 1
under the soft read) and held after the hard transition.  d=3 plane is behind d=2 point here (2 heads); training at
lr 1e-2 is spiky (d3-sh1 dipped 76 -> 50 -> 74).  Single seed: seeds + width 32 next, then the d=3 point mode.

### Frozen embedding hash for the point reads, and the learned head split (job 3132379, `run_zoo_split.sbatch`), width 16, seed 123, 2026-09-11

`hash_src=embed, freeze` = Abdullah's frozen recipe: a fixed random projection of the token EMBEDDING (context-free),
quantised; identical tokens share a cell by construction, nothing to learn, no anneal, no balance penalty.

| read | learned contextual hash (earlier) | frozen embedding hash |
|---|---|---|
| d=3 point (49 cells) | 72.0, collapsed to 32 mid-run | **94.8** (98/94/93/94/95) |
| d=4 point (125 cells) | 42, collapsed to 10 at ep4 | **88.4** (89/80/91/88/95) |
| d=4 heads split c=[1,2], learned hash | | 36 @ ep10, no long recall (killed) |

Reading: the d=3 dip and the fine-c collapses were the learned hash's training (STE quantiser, anneal-time boundary
churn, balance penalty spreading filler), not the geometry: with the frozen hash the point read is the best read at
every d and d=3 reaches 94.8, within 3 of full attention (98) and +48 over Mamba-2 (46.9), in the paper's own
architecture (sparse C, pooled profiles, Theta(1) decode).  The affine family's middle codimension was a workaround
for a bad hash.  Caveat: the frozen embedding hash is exact-token hashing -- right for MQAR (query token == key token),
not for language, where the learned contextual hash is needed and its training is the open problem (fixes to try:
freeze after anneal, gate-weighted balance, learned projection of the embedding, 10x lower hash lr / longer anneal).

### enwik8 proof of concept: content-hashed SMAT vs Mamba-2 on text (2026-09-11, `lm/train_lm2.py`, `lm/eval_pos_loss.py`, `run_lm_poc*.sbatch`)

Bytes, 90M/5M/5M split, vocab 256.  6 layers, d_model 384, headdim 64, d_state 64 (~5.8M non-emb params), T 4096,
batch 8, 1500 steps = 49M bytes, lr 6e-4 cosine.  SMAT arms: paper mask with reset at 2048, content hash (learned,
contextual, 16-byte key-side conv), v10 recipe (gated balance 0.01, anneal 500), point read.  Per-position loss on the
5M test bytes (1220 sequences), window 501.  Fixes needed on the way: Abdullah's point read / pooling materialised
(b h, T_R, N0, p) -- rewritten as bmm (exact, test passes); G branch checkpointed (45 GB -> 19 GB); eval must run a
dummy forward before load_state_dict (lazy per-length hash modules).

| arm | train nll @1500 | val nll | test bpb | nll 0-512 | 512-2048 | 2048-3072 (G on) | 3072-4096 | ktok/s |
|---|---|---|---|---|---|---|---|---|
| Mamba-2 | 1.215 | 1.244 | **1.757** | 1.239 | 1.216 | 1.212 | 1.216 | 607 |
| SMAT d=2 | 1.224 | 1.248 | 1.767 | 1.241 | 1.222 | 1.226 | 1.221 | 94 |
| SMAT d=3 | 1.223 | 1.252 | 1.767 | 1.240 | 1.222 | 1.226 | 1.221 | 59 |

Reading: at this budget the content-hashed G is neutral-to-slightly-negative (+0.01 bpb), with no step at the
boundary where it switches on; Mamba-2's own per-position curve is flat after ~500 bytes, i.e. the model is not yet
using context at all, so there is nothing for G to add.  The test of long-context use needs a trained model:
full budget (4000 steps, d_model 768) or the PG-19 / 16K-token setup.  The G branch costs 6-10x throughput as
implemented (python-level pooling + recompute); a fused kernel is needed before any scaled run.

### Read geometry used by the LM runs (2026-09-11)
Every LM run before this date (enwik8 POC d=2/3/4, the selective-copying script, the first LDC launcher draft) used the
**point read** (`hash_codim = d-1`: each query reads the single pooled cell its own hash lands in).  That was a
side-effect of the OOM fix: the memory-bounded sparse ops (`smat_pool_ops.py`) only implemented the point read.  At d=2
the point and hyperplane reads coincide (hyperplanes of F_q^1 are points), so the d=2 numbers are the paper's mask;
the d=3 and d=4 enwik8 numbers are NOT (they are a hashed cell memory with N0 cells, one cell per query).
`content_addr.py` now runs the affine-subspace read on the sparse path too: pool cells (one bmm), aggregate to the
D * q^c cosets with the incidence M (one index sum over N0 cells, no per-token work), then a point read at
(direction, coset of own cell).  Exact vs the dense plane read in float64 (fwd ~2e-14, grads Phi/Psi/Vb ~1e-14); the
direction logits get a one-sided straight-through gradient (chosen direction only).  Defaults flipped to
`hash_codim = 1` (paper's hyperplane incidence) in `lm/train_lm2.py`, `lm/sc_train.py`, `run_ldc.sbatch` (`arm:d:codim`).
Hard-phase (anneal done) layer cost at width 512 / T 16K / batch 4, single GH200, point read:
mamba2 5.8 ms, d=2 bf16 26.3 ms / 4.5 GB, d=3 bf16 34.5 ms / 7.8 GB, d=4 bf16 58.7 ms / 19.3 GB.

### enwik8 POC rerun: d=3 with the HYPERPLANE read (job 3134208, `run_poc_d3c1_1gpu.sbatch`, 2026-09-11)
Same settings as the d=3 point-read arm (v10, anneal 500, 16-byte key conv, 1500 steps / 49M bytes), read = codim 1
on the exact sparse path (fp32 G branch).  Train loss at step 1500: 1.2228 (point read 1.2233).  Test per-position
loss, 5M bytes, window 501:

| arm | bpb | nll @512 / 1024 / 2048 / 4096 |
|---|---|---|
| mamba2 | 1.7568 | 1.219 / 1.218 / 1.214 / 1.216 |
| smat d=2 (point = plane) | 1.7674 | 1.224 / 1.223 / 1.238 / 1.222 |
| smat d=3 point | 1.7669 | 1.224 / 1.223 / 1.237 / 1.221 |
| **smat d=3 hyperplane** | **1.7647** | 1.222 / 1.221 / 1.235 / 1.219 |
| smat d=4 point | 1.7625 | 1.220 / 1.219 / 1.234 / 1.219 |
| **smat d=4 hyperplane** (job 3134375) | **1.7633** | 1.221 / 1.220 / 1.233 / 1.219 |

Reading: hyperplane vs point is a 0.002 bpb difference at d=3 and 0.001 at d=4, i.e. nothing; every SMAT arm is flat in position and
~0.01 bpb behind Mamba-2, with the same bump at 2048 (the boundary reset).  At this size / budget (10.5M params, 49M
bytes) the model does not use long context, so the read geometry cannot matter yet.  Training throughput on the sparse
path: 187 ktok/s vs 59 ktok/s for the earlier dense point read (3.2x) on one GH200.

### Pooling / read ops: capped sub-row layout (2026-09-12, `smat_pool_ops.py`)
The d=4 hyperplane enwik8 run OOMed (7.4 GB padded tensor) and then crawled at 8-12 ktok/s on the segmented fallback:
real bytes hash with heavy skew (one cell took >1400 of 16K key pairs early on), so the sorted-and-padded layout was
O(B*N0*Lmax) and the segmented index_add serialised on the hot cells' atomics (bf16 -> fp32 accumulation did not
help: 353-489 ms per layer step).  Replaced both by a capped layout: pairs sorted by cell, each cell split into
ceil(count/64) sub-rows of 64 slots, one bmm over sub-rows, one small index_add of the per-sub-row partial sums.
Memory O(pairs + B*N0*64), no per-token atomics, exact (float64 test at caps 64/3/1).  d=4 hyperplane layer at
width 384 / T 4096 / batch 8 in the soft phase (K=8 pairs): 83.7 ms, 5.1 GB (padded 92 ms / 7.7 GB when it fit,
segmented 353 ms).  Training rate at step 50: 46.5 ktok/s vs 11.9 before.

### enwik8 POC at 16K context (job 3134595, `run_poc_len.sbatch` SEQ=16384, 2026-09-12)
Same model / recipe / 49M-byte budget as the 4K arms, batch 2 x 16384 (32K tokens per step), SMAT with the
hyperplane read (c=1), reset at 8192.  Test per-position loss over the 5M test bytes (305 sequences), mean nll by
position segment:

| arm | bpb | pos 0-1k | 1k-4k | 4k-8k | 8k-12k | 12k-16k |
|---|---|---|---|---|---|---|
| mamba2 | 1.7940 | 1.250 | 1.234 | 1.243 | 1.253 | 1.241 |
| smat d=2 c=1 | 1.7974 | 1.251 | 1.235 | 1.244 | 1.257 | 1.244 |
| smat d=3 c=1 | 1.8005 | 1.254 | 1.237 | 1.246 | 1.259 | 1.245 |
| smat d=4 c=1 | 1.7967 | 1.251 | 1.235 | 1.243 | 1.256 | 1.243 |

Reading: all arms within 0.007 bpb, ordering mamba2 < d=4 < d=2 < d=3 (single seed, noise-level).  Loss does NOT fall
with position for any arm: 12k-16k is no better than 1k-4k, i.e. the model uses ~1k of context at this size/budget, so
16K vs 4K only costs (1.79 vs 1.76 bpb, fewer independent sequences per step).  Per-arm training time: mamba2 3 min at
~300 ktok/s, SMAT 5-8 min at 90-150 ktok/s.

### enwik8 POC at 32K context (job 3134596, `run_poc_len.sbatch` SEQ=32768, 2026-09-12)
Batch 1 x 32768, otherwise as the 16K sweep; reset at 16384; test eval over 152 sequences.

| arm | bpb | pos 0-1k | 1k-4k | 4k-8k | 8k-16k | 16k-24k | 24k-32k |
|---|---|---|---|---|---|---|---|
| mamba2-d1 | 1.8162 | 1.268 | 1.248 | 1.254 | 1.256 | 1.255 | 1.271 |
| smat-d2 | 1.8188 | 1.267 | 1.248 | 1.255 | 1.257 | 1.257 | 1.274 |
| smat-d3 | 1.8153 | 1.267 | 1.247 | 1.253 | 1.254 | 1.256 | 1.271 |
| smat-d4 | 1.8172 | 1.269 | 1.248 | 1.255 | 1.255 | 1.257 | 1.272 |

Reading: all within 0.004 bpb (d=3 hyperplane nominally best, 1.8153 vs mamba2 1.8162): noise.  Loss RISES over the
second half of every sequence (24k-32k worst), for Mamba-2 as much as SMAT: with batch 1 the 49M-byte budget is 1500
single-sequence steps, so this is under-training plus enwik8's non-stationarity within a 32K window, not a mixer effect.
4K -> 16K -> 32K costs 1.76 -> 1.79 -> 1.82 bpb for every arm alike.  Conclusion of the whole enwik8 series: at 10.5M
params / 49M bytes nothing separates Mamba-2 from SMAT at any d, read geometry or context length; a real test of the
long-range read needs the PG-19 300M-token setup.

### PG-19 long-context sweep: setup (2026-09-12, `run_pg19.sbatch`, `lm/prep_pg19.py`)
Found: the earlier PG-19 tokenisation had been killed after 5.4M train tokens, leaving a 320M-slot memmap of zeros --
no run had used it.  Re-tokenised: 320M train tokens (book order), val = PG-19 validation split whole (50 books,
4.6M), test = test split whole (100 books, 10.6M), files truncated to their real length.
Model: 8 layers, d_model 384, GPT-2 vocab (19.3M tied embedding).  Non-embedding params: mamba2 7.6M, gdn (fla
GatedDeltaNet, expand_v 2, hd 64, 6 heads) 9.5M, transformer (RoPE attn + 4x GELU MLP) 14.2M, smat d=2/3/4 13.9M
(of which ~3M is the never-used tail of the per-head alpha tables, max_B 65536).  Budget 300M tokens per arm = 9155
steps x 32K tokens, SEQ 16384 (batch 2) and 32768 (batch 1); cosine lr 6e-4, warmup 300; SMAT: hyperplane read, v10
recipe (gated balance 0.01, anneal 1000 steps), 4-token key conv, reset at SEQ/2.
Smoke, each arm alone at 32K (40 synthetic steps, rate includes warm-up so it under-reads): mamba2 64 ktok/s /
27 GB, gdn 14 ktok/s (triton compile dominated) / 29 GB, transformer 178 ktok/s / 29 GB, smat d=2 49 / 32 GB,
d=3 45 / 32 GB, d=4 35 / 32 GB.  Peak is dominated by the 32K x 50257 logits, so arms share a GPU 2 at a time at 32K
and 4 at 16K; the job checkpoints every 250 steps and resubmits itself (one running job per user on the interactive
partition).

### BUG (found 2026-09-12 09:50): the learned content hash was never trained
`SmatMamba2MR` builds its `ContentAssign` modules (hash projection W, gamma, b, direction logits Wd) lazily at the first
forward for each sequence length.  Zoology's `Trainer.fit()`, `lm/train_lm2.py` and `lm/sc_train.py` all built the
optimizer from `model.parameters()` BEFORE any forward, so those tensors were never in the optimizer.  Verified on saved
checkpoints: optimizer had 86 of 116 tensors (enwik8 d=3), gamma still exactly 1 and b exactly 0, W rows at their
randn norm (sqrt(d_model)).  Consequently every "learned contextual hash" number before this date is actually a
**frozen random projection of the hidden state** (contextual but untrained; balance penalty, anneal, hash lr,
freeze-after had no direct effect -- only side effects through the write gate dt in the gated-balance variants and
noise).  Affected: MQAR hash-recipe iteration v0-v11, the affine-family table, the w32/w64 hyperplane sweep below,
enwik8 SMAT arms (all lengths), PG-19 SMAT arms (first pass), selective copying SMAT arms.  NOT affected: frozen
embedding hash (94.8 @ d=3 point, 88.4 @ d=4), all Mamba-2 / GDN / attention baselines, and the kernel/throughput work.
Fix: a no-grad prebuild forward before the optimizer in all three trainers, with an assertion that the optimizer covers
every parameter tensor (prints "optimizer covers N/N (k content-hash tensors)").  Quarantined outputs in
`lm/out/invalid_frozenhash/`, `logs/invalid_frozenhash/`, ckpt/invalid_frozenhash/.  Reruns: `run_mqar_c1.sbatch`
(MQAR w16/32/64 x d2/3/4 hyperplane + w16 d3 point, hash trained), `run_pg19.sbatch` SMAT arms, selective-copying SMAT
arms (running on the PG-19 job's GPU).

### MQAR w32 / w64, hyperplane read, FROZEN-RANDOM contextual hash (see bug above), hd16/ds16, lr 1e-2, 32 ep, seed 123
| width | Mamba-2 (earlier) | d=2 c=1 | d=3 c=1 | d=4 c=1 |
|---|---|---|---|---|
| 32 | 82.8 | **96.2** (64 pairs: 94.3) | 68.7 (15.4) | 74.8 (35.3) |
| 64 | 81.5 | **97.8** (98.4) | 84.2 (48.6) | OOM (GPU shared), rerunning |

Even with an untrained random hash, d=2 (13 cells, hyperplane = point) beats Mamba-2 by +13 / +16 at 64 pairs 94-98;
d=3/4 with the random hash fall off at 32-64 pairs (the random projection does not separate keys well enough for the
finer cells).  The trained-hash rerun is queued (job 3135974).

### Selective copying, d_state 16 (job 3134922, `run_sc.sbatch` DS=16 GDNHD=16), L 4096, 16 tokens / vocab 16, 10k steps
Baselines (valid): Mamba-2 hd64/ds16 **77.1%**, Gated DeltaNet (head_dim 16, same 2048-number state per layer) **96.0%**.
SMAT arms (frozen-random hash, invalid): d=2 44.1, d=3 46.7, d=4 46.2 -- rerunning with the trained hash.  Note the
task structure: the 16 answer-slot inputs are identical blanks, so a content hash of the *query* maps all 16 queries to
one cell; ordered recall by position is exactly what content addressing cannot do, and with the boundary reset the first
half of the prefix is reachable only through G.  Expect SMAT <= Mamba-2 here unless the queries' hash is contextual
(the key-side conv gives the keys context; queries hash their own hidden state, which after the recurrence does carry
position).

### PG-19 300M-token sweep, baselines (valid; `lm/out/pg19-*.json`), nll per GPT-2 token on the PG-19 test split
| arm | 16K nll (ppl) | 32K nll (ppl) | 16K nll @512 / 4096 / 8000 |
|---|---|---|---|
| Gated DeltaNet | **3.678 (39.6)** | **3.738 (42.0)** | 3.698 / 3.640 / 3.685 |
| Mamba-2 | 3.752 (42.6) | 3.803 (44.8) | 3.763 / 3.717 / 3.764 |
| transformer (RoPE + 4x MLP) | 3.797 (44.6) | 3.910 (49.9) | 3.818 / 3.755 / 3.806 |
| smat d=2 / d=3 (frozen-random hash, invalid) | 3.743 / 3.747 | -- | |
GDN leads by 0.07 nats; the transformer is last (33M params, 300M tokens: too little data for attention at 16-32K, and
batch 1 at 32K).  Loss is flat in position for every arm beyond ~1K.  SMAT arms rerunning with the trained hash
(job 3135970 chain).

### MQAR with the hash actually trained (job 3135974, `run_mqar_c1.sbatch`), hyperplane read, v10, hd16/ds16, lr 1e-2, 32 ep, seed 123
Accuracy overall (4 / 8 / 16 / 32 / 64 pairs).  Mamba-2 rows from the small-state sweep.  "random" = the frozen-random
contextual hash numbers from before the fix, same config.

| width | Mamba-2 | d=2 c=1 trained (random) | d=3 c=1 trained (random) | d=4 c=1 trained (random) |
|---|---|---|---|---|
| 16 | 46.9 | 78.9 (84/75/79/81/76) (86.1) | 58.7 (91/81/63/40/18) (67.6) | 73.5 (86/79/82/64/57) (65) |
| 32 | 82.8 | **95.9** (99/99/94/94/94) (96.2) | **94.2** (99/98/92/94/88) (68.7) | OOM, rerun queued (74.8) |
| 64 | 81.5 | OOM, rerun queued (97.8) | OOM, rerun queued (84.2) | 89.6, best 94.9 (100/100/93/83/72) (OOM) |

Reading: training the hash is what makes d=3 work at width 32 (68.7 -> 94.2, 64-pair 15 -> 88); d=2's hash is one
number per token so random was already enough (96 either way).  At width 16 the trained hash is *worse* than random for
d=2/3 (79 vs 86, 59 vs 68) -- the width-16 model has 16-dim hidden states to hash from and the v10 anneal/balance was
never actually tuned (its "tuning" happened with a frozen hash); single seed.  Four runs OOMed (10 runs + 3 selective-
copying arms on one GPU) and are queued (job 3135994's successor).  Frozen-embedding point read (94.8 @ w16 d=3) is
still the best width-16 number.

### Selective copying, d_state 16, final (trained hash; L 4096, 16 tokens / vocab 16, 10k steps, batch 32)
| Mamba-2 | GDN | SMAT d=2 c=1 | SMAT d=3 c=1 | SMAT d=4 c=1 |
|---|---|---|---|---|
| 77.1 | **96.0** | 40.8 | 25.4 | 45.8 |

SMAT is well below Mamba-2 here (frozen-random hash gave 44 / 47 / 46 -- the same).  As predicted from the task
structure: the 16 answer-slot queries are identical blank tokens, so their content hash cannot address 16 different
cells, and the boundary reset hides the first half of the prefix from the recurrence.  Selective copying is a
positional/ordered-recall task; the content-addressed G is the wrong tool for it, and the reset actively hurts.  The
no-reset (hybrid) SMAT would be the fair variant for this task, not run.

### MQAR, hash trained -- complete table (jobs 3135974 + 3136241), hyperplane read c=1, v10, hd16/ds16, lr 1e-2, 32 ep, seed 123
Overall accuracy (64-pair slice).  Mamba-2 from the small-state sweep.  Point read at w16 d=3 for reference.

| width | Mamba-2 | d=2 c=1 | d=3 c=1 | d=4 c=1 | d=3 point (w16 only) |
|---|---|---|---|---|---|
| 16 | 46.9 | 78.9 (76) | 58.7 (18) | 73.5 (57) | 79.8 (95; 4-pair 91, 8-pair 75, 16-pair 61) |
| 32 | 82.8 | **95.9** (94) | **94.2** (88) | **95.4** (94) | |
| 64 | 81.5 | **97.7** (98) | **96.8** (96) | 89.6, best 94.9 (72) | |

Reading: at widths 32 and 64 every d beats Mamba-2 by +12 to +16 overall and by +50 or more on the 64-pair slice
(Mamba-2's 64-pair accuracy is ~40 / ~35), and d=3/4 are now level with d=2 -- the "performance does not decrease with
d" claim holds once the hash is trained (with the random hash d=3/4 fell off at 32-64 pairs).  Width 16 is the odd one:
the hidden state is 16-dim, the hash has little to work with, and the trained point read shows the inverted-slices
pattern again (64 pairs 95, 4 pairs 91, 8-16 pairs 61-75).  The width-16 recipe was never actually tuned; single seed.

### Selective copying, d_state 16, no-reset (hybrid) SMAT arms, final (trained hash)
| Mamba-2 | GDN | SMAT d=2 | SMAT d=3 | SMAT d=4 | SMAT reset d=2 / d=3 / d=4 |
|---|---|---|---|---|---|
| 77.1 | 96.0 | 54.5 | 34.9 | 65.0 | 40.8 / 25.4 / 45.8 |

Removing the reset helps (+14 / +10 / +19) but every SMAT arm stays below Mamba-2, whose recurrence it contains.
Correction to the first reading: identical blank query tokens do NOT make the read blind -- a cell is an r x p
associative memory (sum_j psi_j v_j^T) read by the query vector phi_i, which differs across the 16 slots through the
recurrent context, so one shared cell can still return 16 different values (Mamba-2's single state does exactly this).
What "all queries hash to one cell" means is only that G's partition into N0 cells buys nothing here, so SMAT cannot
beat Mamba-2; it does not explain being BELOW Mamba-2.  Unverified candidates for the deficit: slower optimisation
(every arm was still rising at 10k steps; the STE hash + soft->hard switch at step 1000), the ungated additive read being
noise while the cells are still disorganised, and clutter from noise tokens written into the cells.  Left as a
negative result at the user's request.

### PG-19 16K, trained-hash SMAT arms (job 3135994), nll per token on the test split
| arm | nll (ppl) | @512 | @4096 | @6144 | @8000 (just before the 8192 reset) |
|---|---|---|---|---|---|
| GDN | **3.678 (39.6)** | 3.698 | 3.640 | 3.687 | 3.685 |
| smat d=2 c=1 | 3.743 (42.2) | 3.748 | 3.702 | 3.748 | 3.813 |
| smat d=3 c=1 | 3.744 (42.3) | 3.749 | 3.704 | 3.750 | 3.806 |
| Mamba-2 | 3.752 (42.6) | 3.763 | 3.717 | 3.757 | 3.764 |
| transformer | 3.797 (44.6) | 3.818 | 3.755 | 3.800 | 3.806 |

SMAT edges Mamba-2 by 0.008 nats overall (and by ~0.015 away from the boundary) but pays ~0.05 nats in the window just
before the reset at 8192, and stays 0.065 behind GDN everywhere.  Mid-run train losses had SMAT level with GDN; the
final test gap says otherwise.  The frozen-random-hash SMAT arms scored 3.743 / 3.747 -- the trained hash changed
nothing measurable on PG-19 at this scale.  d=4 (16K) and all 32K SMAT arms still to run (chain job 3136709).

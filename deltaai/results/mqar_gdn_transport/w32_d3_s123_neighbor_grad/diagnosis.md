# Neighbor-gradient transport run: epoch-6 diagnosis

The run is paused and all GPU tasks are stopped. The latest checkpoint and
history are local and on Modal. The normal-mode GPU probe completed before
shutdown and used the first 128 cached validation examples in each of five
cells (640 examples total). After stopping, analysis used saved outputs and
local checkpoint arithmetic; no optimizer steps or GPU jobs were run.

## Matched-epoch performance

| Model at epoch 6 | Overall | 16 pairs | 64 pairs |
|---|---:|---:|---:|
| GDN | 64.48% | 82.24% | 8.01% |
| Original SMAT d=3 | 45.24% | 38.47% | 4.02% |
| Transport, pruned write gradient | 50.03% | 55.65% | 2.60% |
| Transport, neighboring write gradient | 29.63% | 21.14% | 0.89% |

The new model led after epoch 1, but then stayed around 26–30% accuracy through
epoch 6. Validation loss stayed around 5.06–5.12 after epoch 2. This is a
learning plateau/regression relative to controls, not evidence of a NaN crash.
The experiment is incomplete and single-seed; these are early results.

## 1. One first-layer head loses its distant memory

The aggregate GPU probe reported scalar write-to-query retention around 0.5.
That average hides two very different heads. Reconstructing the first-layer
embedding, RMSNorm (epsilon 1e-5), and forget gates from saved weights and the
same cached examples gives the following on the 64-pair cell:

| Checkpoint | Head 1 mean per-token alpha | Head 2 mean per-token alpha | Head 1 mean target-path retention | Head 2 mean target-path retention |
|---|---:|---:|---:|---:|
| New, epoch 3 | 0.999995 | 0.314890 | 0.999315 | 7.26e-34 |
| New, epoch 6 | 0.999997 | 0.267012 | 0.999626 | 4.33e-40 |
| Previous transport, epoch 23 | 0.999994 | 0.957002 | 0.999225 | 0.01059 |
| GDN baseline, epoch 32 | approximately 1 | 0.999998 | 0.999982 | 0.999748 |

The comparison checkpoints differ in training age; they provide context, not
a matched-epoch intervention proving causality. Within the new run, the severe
forgetting is already present at epoch 3. All sampled distant query paths for
head 2 have scalar retention below 1e-6 across every evaluation cell; even on
16 pairs its epoch-6 average is only 3.12e-12.

For normalized keys and beta in [0,1], the operator norm of I-beta*k*k^T is at
most one. The transported coefficient's magnitude is therefore bounded by
beta_j times the scalar product of alphas along the path (before value/output
scaling). An almost-zero scalar product cannot be recovered by choosing a
better hyperplane. This head remains able to process recent tokens; the finding
concerns its distant memory, not the entire head or half the entire network.

The original delta-profile SMAT's local GDN also learns short memory in some
heads, but its distant profiles do not inherit that scalar decay. Its local
gate statistics therefore cannot be interpreted in the same way as transport.
The previous no-decay intervention concerned the old transport checkpoint at
epoch 13. Its negligible effect cannot be generalized to this new checkpoint.

Raw calculations and reproducible analysis are in `head-decay-analysis.json`
and `analyze_head_decay.py`. These reconstruct only first-layer gates; they do
not run a CPU model or reproduce the full network.

## 2. Improved occupancy has not produced target-selective reads

The new epoch-6 probe uses 26.38/49 value buckets in layer 1 on 64 pairs, with
6.76 other-value collisions per value. Layer 2 uses 16.78/49, with 8.08 other
collisions. The older epoch-16 probe used only 4.72 and 7.76 buckets, with 43.59
and 35.11 other collisions. This is better spread, though checkpoint ages
differ and spread alone is not retrieval accuracy.

The new model's geometric reader is essentially unselective with respect to
the correct value-position write:

| Epoch-6 probe | Values covered by 4 reads | Coverage expected for a uniformly chosen value | Actual target coverage/head | Target share of routing weight | Uniform target share |
|---|---:|---:|---:|---:|---:|
| 16 pairs, layer 1 | 14.67/16 | 91.68% | 91.38% | 6.23% | 6.25% |
| 64 pairs, layer 1 | 36.88/64 | 57.63% | 57.87% | 1.562% | 1.5625% |
| 64 pairs, layer 2 | 40.59/64 | 63.42% | 63.27% | 1.601% | 1.5625% |

Target routing share is geometric incidence/selector weight normalized over
actual value positions, before signed transported Q/K coefficients. It is not
the final attention weight. Averaging heads can also hide their relative utility.

Thus, spreading writes makes the four reads exclude more values without
preferentially retaining the right one. Layer-1 any-head target coverage on 64
pairs is 83.74%. But missing coverage is not a complete explanation: on 16 pairs,
any-head coverage is 99.71% while probe accuracy is only 21.00%.

Content matching is also weak. On 16 pairs the actual value-position write has
the largest transported content score only 14.80% of the time in layer 1. On
64 pairs it is 4.43%; ranking by absolute score gives 4.02%, falling to 2.81%
after geometric weighting. These are head-averaged position-level diagnostics;
causal convolutions can encode a pair's information at neighboring positions.
They establish weak selectivity, not an exact decomposition of prediction errors.

## 3. What the gradient change does and does not establish

GPU checks passed against dense forward/backward references, including payload
gradients being counted once, and identical hard outputs at fixed weights.
These checks support correctness of the implemented surrogate. They do not
establish that it is the best optimizer direction for the discrete task.

The added surrogate signal reaches both hash parameters and the shared hidden
states feeding the hash. In layer 2 it can therefore change earlier GDN layers;
the earlier gates are not isolated from the change. The writer can also change
its bucket map while the independent reader still only learns through its
currently selected top-four planes. Its unselected plane logits have no direct
task gradient through the discrete selection.

The most plausible interpretation is an unfavorable joint learning trajectory:
storage spreads out, the reader/content features do not learn correspondingly
selective retrieval, and one first-layer head settles into very short memory.
This explains concrete limitations of the saved checkpoint. It does not prove
which gradient interaction initiated them, nor that LR0.01 alone caused them;
we do not have per-step gradient norms or a controlled retraining comparison.

The earlier diagnosis overemphasized hash collapse as the main fix. Restoring
neighbor terms is not sufficient to solve this configuration. The next useful
causal diagnostic would be a no-training scalar-decay ablation on this new
checkpoint, separately from routing changes. A later training comparison could
isolate the hash surrogate from shared backbone inputs while leaving hash and
causal-hash-convolution parameters trainable. Neither has been run or implemented
as part of this analysis; no new architectural change is justified as a proven fix.

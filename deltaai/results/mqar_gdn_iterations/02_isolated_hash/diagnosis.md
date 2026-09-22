Training is paused at the user's request. Trial 02 automatically rejected itself
after epoch 4 and saved its resumable checkpoint before any epoch-5 training.
All training apps are stopped. A separate evaluation-only GPU probe finished;
it performed no backward passes or optimizer steps. No new candidate was launched.

Further saved-weight analysis in `address-alignment.md` identifies inherited
fixed-position anchor writes and a first-layer preference for their values on
the 16-pair cell. It also shows that applying the writer hash to current query
features does not yet produce aligned addresses. The proposed shared-address
reader needs joint learning, not merely a change of inference-time selection.

The current model is still learning, but its learned routing does not reliably
prioritize the requested item. The overall deficit is concentrated in short,
dense recall. This is a narrower diagnosis than the previous run's severe scalar
forgetting and plateau: cross-boundary scalar forgetting is disabled here.

Full validation, matched epoch 4 and seed 123:

| Key/value pairs | Sequence length | GDN | Current SMAT | Difference (pp) |
|---|---:|---:|---:|---:|
| 4 | 64 | 99.50% | 96.38% | -3.13 |
| 8 | 64 | 96.63% | 88.73% | -7.90 |
| 16 | 64 | 73.19% | 55.54% | -17.65 |
| 32 | 128 | 27.28% | 33.37% | +6.09 |
| 64 | 256 | 6.70% | 9.31% | +2.61 |
| Overall | Equal mean of five cells | 60.66% | 56.66% | -4.00 |

The 16-pair cell alone contributes -3.53 percentage points to the overall gap.
It is therefore misleading to describe this particular result as simply failing
on the longest sequences or greatest number of pairs.

The last change isolated routing-related gradients from shared backbone inputs:
both the write-hash input and the beta weights used only by the occupancy loss
were detached. Hash parameters and the causal hash convolution remain trainable;
memory payloads still train the actual beta gates. This preserves the forward
function at fixed weights. At epoch 4, this raised overall validation accuracy
from 42.25% to 56.66% under the same recipe. That supports gradient interference
as a problem in the prior trial, but does not separate the task-surrogate and
occupancy-loss contributions. It is a single-seed comparison.

The latest accuracy progression was 0.52%, 47.91%, 52.15%, 56.66%; validation
loss fell 7.85, 3.42, 3.17, 2.81. These logs do not show the previous plateau or
accuracy regression. They do not establish eventual convergence: the required
epoch-4 gate stopped training while accuracy was still rising.

Checkpoint interventions used the first 128 cached validation examples per cell
(640 total), with the trained weights frozen. These subset scores should not be
substituted for the full-validation comparison above.

| Evaluation intervention | 16 pairs | 32 pairs | 64 pairs |
|---|---:|---:|---:|
| Normal model | 54.83% | 33.86% | 9.35% |
| Disable cross-boundary memory | 0.00% | 0.02% | 0.00% |
| Remove rank-one transport | 34.77% | 27.00% | 4.60% |
| Replace routing by uniform mean of all planes | 48.05% | 21.17% | 5.00% |
| Force four reads through the target value-position bucket | 64.65% | 28.74% | 17.19% |

Transport is useful to this checkpoint: removing it loses 20.07 pp on 16 pairs.
Disabling memory loses essentially all cross-boundary recall. Simply averaging
all profiles also hurts the dense cases; more coverage alone is not sufficient.
The all-plane mean is computed as the sum of all bucket matrices divided by q,
which is exact for this incidence geometry because every bucket belongs to D
out of D*q planes. It is only a diagnostic and violates the four-read inference
budget; it is not a proposed model.

The label-informed routing intervention improves 16-pair accuracy by 9.81 pp
and 64-pair accuracy by 7.84 pp. It establishes sensitivity to read selection,
but is not deployable or an upper bound. It worsens the 32-pair cell, and it
targets value positions even though causal convolutions can distribute useful
pair information across neighboring positions. It cannot assign an exact
fraction of all errors to routing.

The internal routing measurements explain why selection remains suspect:

* At 16 pairs, the correct value position receives 6.17% of the geometric
  routing weight in layer 1 and 6.28% in layer 2, versus 6.25% for a uniformly
  selected value. At 64 pairs the corresponding figures are 1.561% and 1.569%,
  versus 1.5625% uniform. These are pre-content, incidence/selector weights,
  not final attention or output contributions.
* On 16 pairs, the four reads cover 9.34 values per head in layer 1 and 12.15
  in layer 2. Any-head target coverage is already 100% and 97.17%, respectively.
  Thus, the issue includes poor weighting and interference among included
  values, not just whether at least one head includes the correct bucket.
* In layer 2, the target value has the largest absolute transported content
  score 43.41% of the time on 16 pairs. After geometric weighting this falls
  to 39.97%. Content matching is learning; the geometric selection does not
  consistently reinforce it. These are position-level diagnostics, not
  token prediction accuracies or a full causal decomposition.
* The 16 values occupy only 4.58 of 25 available buckets per head in layer 1
  and 6.78 in layer 2. A value shares its bucket with 7.19 and 5.20 other
  values on average. First-layer hash-coordinate cosine similarities at
  length 64 are 0.931 and 0.941, consistent with correlated coordinates.
  Marginal coordinate balance does not ensure a well-spread joint hash.

There is also a concrete architecture/training-distribution mismatch worth
distinguishing from a broken dataset. The model has separate learned hash and
categorical reader parameters for each sequence length. The length-64 router
trains only on 4-pair examples, then evaluates on 4, 8, and 16 pairs. The dense
16/32/64-pair training examples use length 256 and different routing modules.
Shared backbone parameters can transfer between lengths; the routing parameters
do not directly share those updates. This is a plausible explanation for weak
density generalization, not a causal result from the current ablations. Baseline
GDN faces the same data distribution with shared recurrent parameters.

The best-supported next design question is how to make query selection agree
with the writer's content address, while retaining one hard write and four
distinct weighted plane reads. Currently W_read is independent of the write
hash, and ordinary top-k backpropagation gives unselected plane logits no direct
task gradient. Shared address-based read scoring and sharing routing features
across lengths are task-general candidates for analysis. Neither is implemented
or proven by these results, and no further training should start while paused.

Evidence: `w32-d3-history.jsonl`, `baseline.jsonl`, `checkpoint-probe.json`,
`checkpoint-probe.log`, `parameter-analysis.json`, `probe-manifest.json`,
`SHA256SUMS`, and the archived probe/architecture sources in `source/`.

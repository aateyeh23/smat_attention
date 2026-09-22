# Joint context–key recall

Status: development is complete for the primary d=3 comparison; the requested
all-d sweep and independent confirmation are still in progress. Development
validation selected this experiment. These are not independently confirmed results.

## Task

Each example contains a fresh table of associations `(context, key) -> value`.
The same set of keys appears in every context, so a key alone does not uniquely
identify a record. Information records are serialized as `(context, key, value)`.
After a separator, queries are serialized as `(context, key, PAD)`, and the model
predicts the stored value from its hidden state at the query's key position.
Every binding is queried once. Record order and query order are independently
shuffled. All answer slots in the input are masked; no previous answer is supplied.

Contexts are sampled without replacement from 32 context IDs. The shared key set
is sampled from 512 key IDs. Values are independently sampled from 16 value IDs
for every binding and every example, giving uniform guessing accuracy of 6.25%.
The full vocabulary has 563 tokens, including three special tokens. This is a
retrieval task requiring a joint identifier, not multi-hop reasoning or a test
of generalization to withheld context–key combinations.

| Contexts | Keys per context | Bindings | Sequence tokens | Training examples |
|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 64 | 36,000 |
| 2 | 8 | 16 | 100 | 36,000 |
| 8 | 16 | 128 | 772 | 36,000 |
| 16 | 16 | 256 | 1,540 | 36,000 |
| 32 | 16 | 512 | 3,076 | 36,000 |

All sizes are interleaved during training, for 180,000 fixed training examples.
There are 2,000 development-validation and 4,000 development-test tables per cell.
The independent confirmation will contain 10,000 fresh tables per cell, shared
by every evaluated model. The data seed is 20260922; split codes are 1001, 2001,
3001, and 9001 respectively. Data audits verify decoding, masks, paired-key
construction, file hashes, and example disjointness. Fresh tables use the same
identifier distribution as training.

## Models and training

Compare the existing native Mamba-2 and Gated DeltaNet implementations with their
SMat d=2,3,4 variants, all with two layers and width64. Use seed123, batch256,
AdamW, weight decay0.1, BF16 autocast, and a cosine learning-rate schedule over
32 epochs. The common-recipe dimension sweep uses LR0.003. There are 705 updates
per epoch and 22,560 updates per run, with 5.76 million example presentations.
All models receive identical cached data and batch orders. There is no
accuracy-based early stopping; final-epoch checkpoints are the primary results.

The architectures are frozen under `source_v5`. GDN's SMat memory retains its
original scalar decay and delta-transition transport. The Mamba SMat memory uses
its original independent write gate without extra-memory scalar decay; its native
Mamba branch retains its normal dynamics. Separately labeled earlier no-decay
ablations are not substituted into this study.

Backbone dimensions and exposure are matched, but total parameters and training
compute are not. Report total and trainable parameter counts for every variant.
The existing SMat implementation prebuilds routing modules for the five sequence
lengths; this study does not evaluate length extrapolation or isolate the mask's
effect from the additional learned parameters.

Native LR0.01 controls are included for both backbones, as is a separate d=3
Mamba LR0.01 optimization trial. Keep these development results visible. The
all-d sweep uses common LR0.003 SMat checkpoints. Native GDN LR0.003 has the
strongest completed macro score, while LR0.01 is stronger at high load; fresh
confirmation must include both. This yields eight primary checkpoints plus one
additional native control.

## Measurements and interpretation

Report per-query accuracy and exact-table accuracy for every load, alongside
equal-cell macro accuracy. Include key-only and context-only majority controls:
each knows the stored table but predicts one modal value for every group sharing
its selected identifier. The conservative single-field control chooses the better
of these two rules for each entire table. Its 512-binding validation score is
19.2249%. Beating it supports retrieval beyond these single-field rules; the
control is not a causal ablation of the model's internal mechanism.

The primary d=3 development results are positive for both backbones. Against the
strongest native setting for each metric, Mamba gains 10.93 percentage points in
macro accuracy and 7.79 at512 bindings; GDN gains 13.67 points in macro and14.30
at512 bindings. Both clear the single-field control. The common-LR curves have
larger gains at128 bindings than at512, so the data do not support a monotonically
increasing advantage with memory load. The one-context cell is effectively
ordinary key recall and provides an easy reference point.

Fresh confirmation must lock every checkpoint before generating tables, retain
all d=2/3/4 outcomes, and evaluate the same tables for every model. Paired bootstrap
intervals resample whole tables (2,000 draws), independently within each cell for
macro scores. The prespecified primary confirmation criterion is positive d=3
SMat-minus-native macro and512-binding differences, and a positive largest-cell
SMat-minus-single-field difference, with 95% intervals excluding zero in both
backbones. The additional GDN control must also be beaten. Added-d intervals are
descriptive; do not select a new winner after observing fresh results. These
intervals quantify evaluation sampling conditional on one training seed.

Earlier negative block-context, large-only, and smaller mixed-size studies remain
in `report.md` and `TRIALS.md`. In particular, SMat Mamba lost the completed smaller
mixed-size comparison. The expanded study changes the identifier vocabulary and
training mixture as well as maximum load, so its128-binding scores cannot be
spliced into the earlier curve. Any paper claim must identify the development
search and the limits of a single-seed, backbone-width-matched comparison.

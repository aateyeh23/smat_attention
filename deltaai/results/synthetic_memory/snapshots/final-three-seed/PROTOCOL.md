# Synthetic memory protocol

Two-layer Zoology Mamba2 blocks, width 32 for development and width 64 for the
conditional main sweep. Native Mamba2 and GDN mixers and the existing MQAR
one-write/four-read SMat variants are reused. Head and state dimensions remain
16 within each backbone family; Mamba2 expands to 2x width, while GDN uses two
heads. SMat retains the existing halfway recurrence reset and content routing.
The input generator does not inspect the SMat mask or boundary.

State tracking uses 256 possible identifier tokens, 16 equiprobable state tokens,
and PAD/QUERY tokens. Every example chooses distinct active identifiers anew,
assigns exactly three independent updates to each, shuffles all update records,
and places them in randomly chosen two-token slots before four distinct random
queries. Targets are the last updates in actual sequence order. No answer tokens
are fed into the queries. No absolute positional embeddings are used. Sequence
length is rounded up to a multiple of 64, with at least 32 slack tokens (and a
minimum length of 256); it grows when the updates no longer fit. Thus memory
load and sequence length covary at larger loads and this is a limitation of the
primary sweep, not evidence for an isolated memory-capacity effect.

Training data are generated afresh using a separate seed stream per step,
independent of all model RNG use. Validation and final test use independent seed
streams. All models within a comparison receive identical training and evaluation
arrays. AdamW, LR 0.01, weight decay 0.1, gradient clipping at 1.0, cosine LR to
zero, BF16 autocast, batch 32, and the existing SMat routing auxiliary loss with
coefficient 0.01. Baselines have no routing auxiliary loss. There is no task-specific
extra input or loss for SMat. Metrics use unrestricted full-vocabulary predictions
at query positions; reference value chance accuracy is 1/16 = 6.25%. State exact
accuracy requires all four predictions to be correct.

First stage: 200 steps, seed 123, Mamba2/GDN with d=1 (baseline) and d=3, loads
16 and 64. Validation every 50 steps on 256 examples; final test on 1,024 fresh
examples. Later stages are conditional on observed learning, as requested; no
expensive sweep is launched before real tiny training has succeeded. All negative
and tied results are retained. Any global calibration will be documented here.

Outputs per run: recipe.json, metrics.jsonl, status.json, checkpoint.pt (model,
optimizer, schedule and RNG), result.json. collect_synthetic_memory.py produces
CSV, Markdown tables, PNG/PDF load plots, and PNG/PDF learning curves.

Variable tracking uses 512 identifier tokens and 16 value tokens. H=2 means two
variable-to-variable links plus a terminal value assignment: x→y→z→VALUE. The
3*N variable identifiers are sampled without replacement, so chains cannot
intersect. Every variable has exactly one outgoing assignment. All assignments,
including terminal ones, are shuffled together and placed into random pair slots;
a uniformly sampled root is queried. Assignments denote declarative links, not
imperative copies of the current value. Length uses the same rounding/slack rule
as state tracking (one query instead of four). This is composition of stored
relations, unlike direct MQAR; unrestricted random assignment order can be hard
for a two-layer causal model even at low load. It is not claimed that failures
isolate memory capacity rather than compositional depth. Initial loads: 8 and 32.

Calibration decision 1: all four width-32 models at 16 entities scored 5.71–5.83%
independent-test accuracy after 200 steps, with zero exact sets and validation
loss near log(16). Repeat both state loads (16,64) from the same initial seed
with 1,000 total steps and the correspondingly extended cosine schedule for all
four models. Keep every other task/optimizer setting fixed. This stage is named
`tiny1000`; the original 200-step results remain intact.

A small post-training data audit independently reconstructed labels from token
records for eight examples at each initial task/load; all latest-state and chain
terminal answers matched, and chains had exactly three distinct variable nodes.

Calibration decision 2: all eight variable-tracking cells completed 200 steps
at 5.08–6.93% test accuracy, near chance. Repeat both loads (8,32), H=2, with
1,000 steps for all four models and no other changes (`tiny1000`).

Evaluation-only controls added: most frequent context value (ties choose the
lowest token ID) and last context value, repeated for every query. These ignore
the queried identifier and can exceed 1/16 because some answer values repeat.
They do not change training or benchmark generation. They were backfilled for
the first pilots by regenerating final-test arrays and checking their saved
SHA-256 digests. Gains over value chance alone do not establish entity-specific
retrieval or successful chain composition.

Calibration decision 3: all eight state cells remain at chance after 1,000 steps
(5.52–5.74% at 16 entities; all four 6.74% at 64 entities). The easy calibration
uses 4 and 8 entities and a minimum sequence length of 64 instead of 256, still
three updates/entity, four distinct random queries, random gaps/order/identities,
1,000 steps and the identical optimizer for every architecture. Actual lengths
are 64 and 128. This is `compact1000`; earlier results are retained unchanged.
The planned high-load sweep will not be used to claim a memory-capacity result
unless calibration first shows a meaningful retrieval signal.

Calibration decision 4: the complete H=2 variable pilot at 1,000 steps remains
at chance (5.37–7.13%) at both original loads. Apply the same compact calibration:
2 and 4 chains, minimum length 64 (both cells have length 64), H=2 unchanged,
1,000 steps, batch 32 and the same LR/optimizer across all four models. No
architecture-specific task or optimization changes are permitted.

For compact and later runs, `query_blind_oracle_accuracy` additionally reports
the most frequent *final* entity state (computed exactly), or most frequent
terminal chain value. It has full task-state knowledge but ignores the query;
it is a stronger control for whether accuracy reflects query-specific retrieval.

State calibration outcome: plain GDN reached 89.11% at four entities (60.94%
exact sets), and 36.06% at eight entities, exceeding the respective query-blind
oracle controls (33.42% and 25.71%). Mamba2 and the tested SMat variants mostly
remain near chance. This is a clear *negative* initial SMat comparison and is
retained. The compact encoding is now fixed; no further state-task changes will
be made to favor a model. The second pilot will cover loads 4,8,16,32, all eight
models, width 32, seed 123, with a uniformly increased 3,000-step budget to inspect
learning curves. LR 0.01, batch 32, all other optimizer/task settings unchanged.
Validation every 250 steps on 256 examples and test on 1,024 independent examples.

Variable calibration: at two chains, Mamba2 reached 44.43% and GDN 16.89% after
1,000 steps, but the query-blind control is 54.98%. This demonstrates learning
of context values, not yet chain following. The encoding and H=2 are now held
fixed. The second pilot will uniformly extend training to 3,000 steps for all
eight models at 2,4,8,16 chains (width 32, seed 123, same LR/batch/optimizer).
All model gains will be interpreted relative to both matched backbone accuracy
and query-blind controls; above-chance accuracy alone is insufficient.

User steering: prioritize tasks where SMat beats its matched backbone at **500
steps**. The pending 3,000-step pilot (3169928) was canceled before starting.
The next stage is now `screen500`: all eight models, width 32, one seed (123),
state loads 4,8,16,32 and variable loads 2,4,8,16; exactly 500 training steps,
cosine schedule over 500 steps, otherwise the fixed compact recipe. Validation
uses 1,024 examples every 100 steps and the final independent test uses 4,096
examples. All four d groups run on separate GPUs in one DeltaAI interactive job.
These are matched comparisons; no variant receives a different budget.

The saved step-500 validation measurements of the earlier 1,000-step runs are
only provisional screens (their LR schedule was different). They show a GDN
SMat d3 gain at two chains, 12.89% vs 6.25%, but below the query-blind control;
state tracking strongly favors plain GDN. Tiny differences near chance are not
credible retrieval advantages. Longer training is deferred under the new goal.

Launch fix: parallel job 3169956 exited before training because argparse treated
`-d1` as an option instead of a completion-tag value. Changed the shell argument
to `--completion-tag=-d1` (and analogously for other d). No model/task changes.
Canceled its unsatisfiable dependent job and resubmitted the same 500-step runs.

Concrete geometry fix: the existing d4 constructor rejected length 192 because
its native prime selection gives 125 profiles but only 96 landmark positions.
Keep the architecture and requested d unchanged; round **192 to 256 for every
model**. No mask-dependent labels, keys, update positions, or query choices are
introduced. This is a shared input-compatibility padding rule. Lengths
64,128,256,448,832 were checked with the existing mask constructor and retain d4.
The six completed state/load16/T192 cells and failed d4 initialization were
preserved under `../synthetic_memory_archive/screen500-state-n16-T192/` and excluded
from the comparison. All eight state/load16 models are rerun at T256; remaining
completed loads are unchanged and reused. Variable/load16 and future main runs
use the same common padding rule. Job 3169963 ended on this error; its blocked
dependent was canceled and both screens resubmitted. This is not an outcome-based
benchmark change.

Queue adaptation: the corrected four-GPU rerun was estimated to wait over an
hour. Scheduler estimates put a single interactive GPU about two minutes away.
Canceled the pending four-GPU jobs and resubmitted the same cells sequentially
on one GPU with 20-minute allocations and an 18-minute checkpointed time slice.
Completed cells are reused; recipes, data and optimization are unchanged.

The completed width-32 state screen has one clear raw-accuracy candidate:
Mamba2+d2 at four entities, 16.10% vs 6.39% for Mamba2 (+9.71 points). It is below
the query-blind oracle, so this is not yet established as query-specific tracking.
GDN+d2 is 53.39% vs GDN 54.68%; d3/d4 are much worse there. At loads 8–32,
no convincing SMat gain is present. The full screen table and curves were generated.

State main plan under the 500-step objective: width 64, same 500-step recipe,
all eight models, seed123, requested loads 16,32,64,128 **plus low-load anchors
4 and 8** to check the observed small-load candidate. No training or encoding
changes. The job waits for completion of the width-32 variable screen, generates
all pilot plots/tables before starting, and runs two independent d groups per
GPU on two interactive GPUs. The low-load additions are labeled anchors, not
replacements for the full requested sweep. Additional seeds wait until complete
one-seed tables and plots have been inspected.

Main-run evaluation adds a query-specific diagnostic without changing training:
compare each model prediction with the correct queried value and with the value
of a different entity/chain from the same example. State tracking averages over
the other three queried entities; variable tracking averages over all other
chains' terminal values. `binding_gap_pp` is correct-query accuracy minus this
mismatched-query accuracy. Query-independent strategies have expected gap zero,
even if they exploit context-value frequencies. A positive reproducible gap is
evidence of query-specific information; this is useful because beating the
matched backbone while remaining below the query-blind oracle is ambiguous.
Final predictions/targets are saved in test_predictions.npz. This diagnostic is
evaluation-only; task data, optimizer, model and training trajectory are unchanged.

Both width-32 screens completed 64 cells at 500 steps; tables/plots were archived
in `snapshots/screen500-one-seed/` before width64 training. Variable tracking has
no convincing SMat win in this exact schedule (at two chains, GDN+d3 is 12.48%
vs GDN15.50%; Mamba2 is18.73% and all its SMat variants are about6.3%).

For quick width64 feedback, a 10-minute one-GPU interactive slice runs while the
longer two-GPU allocation waits. The interactive partition enforces one running
job per user, so these cannot train the same cell concurrently. The two-GPU job
skips completed cells and resumes unfinished checkpoints. All use the same
`main500` recipes and data. Scheduler/placement changes do not alter training.

The width64 variable comparison is also bounded to exactly 500 steps: all eight
models, requested loads 8,16,32,64,128 plus low-load anchors2,4, seed123. It checks
the originally requested main width before rejecting a task based only on width32;
it does not extend the training budget or retune the task to seek a SMat win.
The full one-seed state/variable tables precede seed confirmations. The largest
loads can remain at chance; those results are retained and are not interpreted
as asymptotic capacity measurements.

During confirmation, completed pilot checkpoints without the new binding metric
are reevaluated on the identical hashed test arrays, without further training.
Their primary saved accuracy/loss/time remains unchanged; the new diagnostic and
`binding_eval_accuracy` are recorded separately. This avoids treating the
original raw-accuracy screen as proof of query-specific retrieval.

Model-selection correction during provenance audit: the initial GDN+SMat runs
used `zoo_smat_gdn.SmatGDNReset` with delta pooled memory, the frozen MQAR width
sweep implementation. However, the repository's selected current recipe is
`zoo_gdn_transport.SmatGDNTransport` with write_hash_neighbor_grad=True,
transport_scalar_decay=True, detach_write_hash_input=True, memory_plant=False,
and memory_incidence_rescale=False (`zoo_gdn_transport_configs.py`). The primary
GDN+SMat comparison must include that selected implementation. It is added as
`gdn_current`; `gdn` results remain explicitly labeled legacy delta-pool controls.
For d1 both names instantiate the identical native GDN baseline, with matched
initialization/data/optimizer. Mamba2 remains the existing four-read implementation.
No architecture implementation or task is redesigned based on the scores.

First run the current GDN baseline and d3 at the small state loads4,8 for500steps,
then the variable counterparts, before its all-d load sweeps. The queued older
variable main sweep was canceled before starting to prioritize this correction.
All completed older results and earlier source snapshots remain available.

The selected current transport recipe completed its small d1/d3 pilots without
errors. Its all-d width32 state screen has no advantage over native GDN:
at four entities, d1/d2/d3/d4 accuracies are54.68/14.59/6.42/33.84%, with binding
gaps42.95/5.55/0.15/19.20pp. Thus some transport variants do learn entity binding,
but worse than the matched backbone. At8–32 entities all SMat scores are near
chance. The current-transport width64 state sweep uses the same six loads as the
completed Mamba/legacy sweep. Pilot tables are regenerated before it starts.

Width64 current-transport state/load4 gives82.62/83.59/85.44% for d2/d3/d4,
with strong binding gaps72.55/74.09/76.64pp. Its d1 rerun gives45.98%, whereas
the previously completed identical native-GDN recipe gives69.87%. Data hashes,
recipe values other than family/output path, and first-step loss/gradient are
identical; later training differs. The existing BF16/custom-kernel pipeline was
not configured for bitwise determinism; the precise source of divergence is
unresolved. Both executions are retained, not counted as independent seeds.
Transport variants beat both native-GDN executions at this setting. Final claims
must include multi-seed results and sensitivity to the alternative seed123 baseline.

All104 primary width64 one-seed cells completed:48 state (loads4,8,16,32,64,128)
and56 variable (loads2,4,8,16,32,64,128), with all eight current models. The complete
tables/plots are archived in snapshots/complete-current-one-seed-before-confirmation/.
At two chains Mamba2+d2 has44.26% raw accuracy versus28.69% for Mamba2, but its
binding gap is-2.81pp versus-1.05pp; this is not evidence of chain following.
At the originally requested variable loads8–128, all models remain near chance.

Final confirmation selection, made after inspecting the complete one-seed sweep:
state width32/load4 (original Mamba d2 candidate), state width64/loads4,16
(current-transport candidate plus original-range floor control), and variable
width64/loads2,8 (raw Mamba d2 gain without binding plus original-range floor).
Use seeds123,456,789 and BOTH primary families with ALL d1/d2/d3/d4 in every
setting:120 cells, of which40 already exist and80 require new500-step training.
This includes negative/tied settings and all losing variants. Seed123 is reused,
with evaluation-only binding backfill for its earlier width32 Mamba runs.
No benchmark or optimizer changes. The optional hop sweep is omitted because
H=2 query-specific chain retrieval has not been established at500steps.

Final confirmation job3170450 completed all120 selected cells (40 reused,80 new)
on two GH200 GPUs in ghx4-interactive. Current GDN+d4 at width64/load4 reaches
83.57±9.08% versus46.08±36.51% for native GDN and wins each observed seed;
using the earlier stronger seed123 baseline raises GDN to54.05±39.00%, without
changing that ordering. GDN+d3 improves the mean; d2 does not. No robust state
advantage appears for Mamba. Variable raw-score candidates do not establish
chain binding and the original d2 gain reverses on average. Final independent
audit verifies354 completed runs,248 primary500-step cells,31 matched comparison
groups,120 confirmation cells, and220 saved prediction files. See FINDINGS.md
and confirmation.md for interpretation, exact accuracy, parameters and all
negative/tied results. No task or optimizer changes were made during confirmation.

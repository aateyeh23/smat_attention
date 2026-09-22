# Joint recall development campaign

Objective: find a reproducible joint-recall setting in which SMat improves over
its native backbone, using one training seed (123). The user permits a negative
outcome in one backbone. Report both families and every requested dimension.
The existing capacity protocol retains its stronger, previously fixed d=3
confirmation criterion for both backbones.

The standalone [dataset description](DATASET.md) contains only the task and data specification.
See [all trials and motivations](TRIALS.md), the periodically refreshed
[validation report](report.md), and [machine-readable results](summary.json).
The expanded [capacity-sweep protocol](CAPACITY_PROTOCOL.md) retains the earlier
mixed-size experiment and extends the load curve through512 bindings.
Failed trials remain in these records. Validation selects configurations; existing
test results are exploratory. A selected positive configuration must subsequently
pass a fresh, independently generated confirmation test.

Each complete run sees 180,000 training examples for 32 epochs (5.76 million
example presentations). Comparisons within a dataset use identical cached examples,
batch orders, model width, layers, and exposure. Tested learning rates, weight decay,
and initialization settings are recorded in each run's `recipe.json`. Compare each
SMat model against the strongest completed native setting on the same dataset.
The original block-context task and all explicit-record variants are distinct
datasets, not interchangeable results. Mixed-size results include every size.

The current dynamic queue is `campaign.json`; live worker status is
`campaign_status.json`. Jobs run on DeltaAI's `ghx4-interactive` partition under
account `bekw-dtai-gh`. `run_joint_recall_iterations.sbatch` processes the queue
for 110 minutes, saving resumable checkpoints. Submit that same launcher again
if the queue remains unfinished; it skips completed runs. There is no automatic
resubmission. The older three-seed continuation launchers are historical.
For the expanded capacity sweep, continuation job3172515 was submitted with an
`afterany:3172269` dependency using the same interactive launcher and dynamic queue.
It resumes incomplete checkpoints and skips completed tasks; check Slurm for its
current state before submitting another continuation. The queue also accepts
explicitly labeled confirmation tasks after a completed positive validation
selection; training-budget audits exclude those evaluation-only tasks.

Backup continuation job3173028 has dependency `afterany:3172515`, using one GPU,
four CPUs, and64GB on the same interactive partition. It can finish evaluation
if the current wrapper defers and resume the added LR0.004 follow-up. It skips
finished queue tasks; cancel it only if all training and confirmation work,
including the independent audit, finishes in the current allocation.

Model source snapshots are immutable under `source` through `source_v5`.
Versions2 and3 add training initialization and optimizer options. Version4 also
records and reconstructs an existing GDN memory option, disabling scalar decay
in the extra memory branch for separately labeled ablations. Native GDN dynamics
and delta-transition transport remain intact. Version5 permits the larger key
vocabulary for the capacity sweep; its model implementations are unchanged.
Each snapshot has a source manifest.

`collect_joint_recall_iterations.py` refreshes the report and learning curves.
`audit_joint_recall_iterations.py` verifies completed budgets, checkpoint selection,
and metrics recomputed from saved predictions. Data-generation audits live beside
the relevant dataset. `confirm_joint_recall.py` implements locked model selection,
fresh confirmation tables, key-only and context-only majority controls, and paired
table-bootstrap intervals. The conservative single-field control selects the better
majority rule for each whole table.
Those intervals describe evaluation uncertainty conditional on seed 123, not
robustness across training seeds.

`audit_joint_recall_confirmation.py` independently decodes every fresh record,
checks split disjointness, reconstructs both majority controls, and recomputes
the saved prediction metrics and paired intervals. Evaluation validity and the
positive-result criteria are reported separately. Fresh runs preserve their
evaluation source alongside the locked model selection.

Mechanism probes are under `diagnostics/`. Their small validation batches diagnose
memory behavior; they are not substitutes for full validation or confirmation.

The user requested d=2,3,4 for both backbones if the capacity task succeeds.
`pending_all_d_tasks.json` records the four new common-recipe commands. The
completed conditional enqueue process passed `all_d_gate.json` and added all
four to the runnable campaign (41 training runs total). Both d=3 variants beat
every native setting in macro and largest-cell accuracy and cleared the
best-single-field oracle. Mamba d=2 began in job3172515; check live status for
subsequent dispatches. All eight variants will share the
same fresh confirmation tables after their final checkpoints are locked. The
additional GDN LR0.01 native checkpoint must also be included because it is
stronger at512 bindings than the macro-selected LR0.003 native checkpoint;
fresh evaluation therefore covers nine checkpoints.

`confirmation_plan.json` fixes the nine model folders before fresh evaluation.
The last queue entry, `confirmation_capacity512_all_d`, runs
`run_joint_recall_confirmation_when_ready.py`: it waits for the selected final
checkpoints, verifies full training budgets and the positive d=3 validation
comparison, then locks checkpoint hashes and generates the fresh split. It
defers cleanly to a later allocation if fewer than15 minutes remain. This
evaluation task is excluded from training-run counts and training-budget audits.

The active confirmation wrapper started with a15-minute reserve. Keep its
source unchanged while it waits. Finished capacity runs took3.4–10.7 seconds
for both saved development-test evaluations (40,000 tables per model combined);
the reserve includes fresh data generation, model initialization, and bootstrap
computation. The wrapper is waiting in job3172515; no fresh tables exist yet.

After prediction, the fresh evaluator invokes its frozen independent audit
and report generator. `result.json` records completed prediction; inspect
`audit.json` and the final process exit status before claiming successful
confirmation. Audit validity and positive d=3 comparison criteria are separate.

At the user's request, `explicit_context_capacity512_lr004` adds native GDN and
SMat GDN d=2 with LR0.004 and the original seed123. Its local protocol fixes the
comparison before training. The dynamic queue now contains43 training tasks
and the original confirmation task. These follow-up runs use the same cached
data and full32-epoch budget, and remain separate from the primary LR0.003 sweep.

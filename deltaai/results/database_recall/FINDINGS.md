# Database-query pilot: no learned selection at 500 steps

The database direction is implemented and eight pilot runs completed on
DeltaAI `ghx4-interactive`. The four models were native Mamba-2, Mamba-2+SMat d3,
native GDN, and GDN+SMat d3. Each used four records, width64, two layers, T64,
seed123 and exactly500 steps. No broader sweep was launched because none
established meaningful conjunction selection in this setup/budget.

This is **inconclusive about a comparative combinatorial advantage**, with no
positive SMat evidence. It is a pilot learning failure, not a successful
baseline-versus-SMat discrimination or evidence of a capacity limit.

## Task

Rows associate an object ID with owner, color and location. Two queries select
objects by either one attribute or a conjunction of two attributes. Each query
lists all object IDs; the model outputs YES/NO membership without receiving
answer tokens as input. Query fields vary, so the same objects can participate
in different overlapping selections. Empty and full sets are allowed. This is
not arbitrary subset labeling or a task constructed from SMat geometry.

All individual attribute values appear in training queries; four of sixteen
value pairs per pair of fields are reserved for held-out conjunction queries.
Rows have no such restriction. Seen and held-out tests have paired databases,
query field choices and candidate order. This tests combinations of known
predicates, not novel object schemas. See [protocol](PROTOCOL.md) for details.

## Results

Conjunction balanced accuracy (%) averages YES recall and NO recall; constant
YES or NO predictions score50%. Each test has4096 databases, two queries per
database, and four membership predictions per query. These are single-seed
results; no uncertainty across training seeds has been estimated.

| Model | Unweighted, seen | Unweighted, held out | Balanced loss, seen | Balanced loss, held out |
|---|---:|---:|---:|---:|
| Mamba-2 | 50.00 | 50.00 | 50.00 | 50.00 |
| Mamba-2 + SMat d3 | 50.00 | 50.00 | 50.00 | 50.00 |
| GDN | 50.17 | 50.23 | 50.00 | 50.01 |
| GDN + SMat d3 | 50.00 | 50.00 | 50.00 | 50.00 |

With unweighted cross-entropy, three models predict NO everywhere. Native GDN
shows weak single-field behavior (52.93% seen balanced accuracy), but negligible
conjunction discrimination. Roughly84% pooled raw accuracy and77% conjunction
exact-set accuracy are explained by the majority/empty-set baselines. No model
correctly predicts a nonempty conjunction set in either test split.

The separately labeled balanced-loss follow-up gives positive and negative
examples equal expected weight for each query arity. It keeps data, optimizer,
architectures and500-step budget fixed for all four models. Predictions mostly
switch to YES, without learning which objects match. Balanced accuracy stays
near50%. This objective change was made after seeing the initial class collapse;
both conditions are retained. It is not a pre-registered benchmark result.

[Object-binding diagnostics](object_binding_diagnostics.json) compare accuracy
against the correct object's label with accuracy against every other object's
label in the same query. The pooled gap is at most0.12 percentage points across
these runs/splits. Thus even small marginal gains are not evidence of learned
object-specific selection. This diagnostic is distinct from the inherited
`binding_gap_pp` field, which records pooled balanced discrimination.

## What this establishes

The task now requires set selection and has an explicit composition holdout,
which were missing from the preceding composite-key recall experiment. However,
the current training setup has not learned the task, including the simple
single-field control. The failed balanced-loss follow-up shows that this one
change is insufficient; it does not identify the cause of the failure.

We did not add records, sweep seeds, tune individual architectures, or extend
training after these results. Before making a mask-expressivity claim on this
task, a common setup needs to demonstrate actual selection learning. SMat
still includes extra parameters/memory, routing loss, and the existing reset;
these runs do not isolate the structural mask from those differences.

## Verification and artifacts

[Audit](audit.json): all eight runs passed independent row/query/label decoding,
holdout-disjointness checks, paired-test checks, regenerated validation/test
hashes, matched recipes, and recomputation of balanced accuracy and exact sets
from saved predictions. The audit checks the complete intended eight-run grid.

[Full tables and plots](report.md), [CSV](results.csv), [job accounting](jobs.txt).
Jobs3171921 and3171939 both completed successfully, using one GPU each for
1m53s and1m57s. Each run preserves recipe, checkpoint, learning curve, final
metrics and raw predictions for both seen and held-out tests. Source archives
and SHA-256 manifests preserve the pilot and final implementation.

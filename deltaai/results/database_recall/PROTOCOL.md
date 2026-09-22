# Database selection: protocol before pilot results

Question: do the existing SMat models improve query-dependent selection of
overlapping sets, including unseen attribute-value combinations, at 500 steps?

Each database contains R objects with unique, randomly sampled object IDs and
three categorical attributes: owner, color, location. Each attribute has four
values, sampled independently and uniformly per object. Attribute domains use
distinct token ranges. Rows are `(ID, owner, color, location)`, shuffled into
random four-token slots with padding gaps. Placement does not inspect any model
geometry or reset boundary.

Each database has two queries in random order: one single-field equality and
one two-field conjunction. Query fields vary uniformly. Queries specify three
attribute slots, using ANY for unspecified fields, then list all database IDs
in random order. Predict YES/NO at each ID. Answers are never supplied as input.
This membership vector represents the entire selected set without introducing
an autoregressive answer-ordering problem. Databases and queries are generated
from semantic predicates, not arbitrary subset labels or incidence geometry.
Empty and full answer sets are allowed; we do not balance or reject databases
to impose a desired selection pattern. The two answer sets can overlap.

For each pair of attribute domains, train conjunction queries use 12 of the
16 possible value combinations (unequal category indices). Four combinations
(equal category indices in the two distinct domains) are held out. All field
pairs and individual values appear in training; database rows themselves are
unrestricted. This is a query-composition holdout, not a holdout of row tuples.
Validation and primary test use the seen-combination distribution with fresh
databases. A second independent-from-training test uses held-out conjunctions;
its database, single-field query, field-pair choice, and candidate ordering are
paired with the primary test via identical random draws. No model selection
or updates use the held-out test. Training/validation/test RNG streams differ.

Pilot: R4, T64, width64, two layers, seed123, native Mamba-2/GDN and SMat d3.
Exactly500 steps, batch32, AdamW lr.01/wd.1, cosine decay, clipping1, BF16;
reuse the existing model implementations and trainer. Validation1024 examples
every100 steps; final tests4096 examples each. Launch immediately, before
extensive tests. Pilot job3171921, one GPU on DeltaAI ghx4-interactive.
Only extend to more variants/seeds/records after inspecting this actual pilot.
No additional training steps or architecture changes.

Primary interpretation: conjunction balanced accuracy and exact selected-set
accuracy, on seen and held-out query combinations. Report single-field and
pooled metrics separately, positive recall, micro-F1, and exact sets restricted
to nonempty targets. Ordinary membership accuracy is secondary: always-NO has
expected75% single-field and93.75% conjunction accuracy. Balanced accuracy is
50% for always-NO. Exact-set results include the corresponding empty-set
baseline, since many uniformly sampled conjunctions have no matches.

Models predict over the entire vocabulary. Balanced accuracy averages exact
YES recall and exact NO recall; invalid tokens are errors. The inherited
`binding_gap_pp` completion field is 200*(pooled balanced accuracy−.5), not the
other-context metric from the preceding experiment. SMat's additional memory,
parameters, routing loss, and existing local-state reset remain part of the
comparison. Matching widths does not match total parameters. Matching seeds
does not guarantee bitwise deterministic BF16/custom-kernel execution.

Pilot result: all four models have conjunction balanced accuracy about50%;
three predict NO everywhere, and native GDN only weakly learns single-field
selection. Raw pooled accuracy about84% is explained by class imbalance.
No architecture-specific advantage is established. Before any larger sweep,
run the same four models/seed/data with balanced loss: for arity k, a YES label
receives weight0.5/(4^-k), NO receives0.5/(1−4^-k); normalize by total batch
weight. This gives equal expected positive/negative weight for each query type.
The task/distributions and500-step budget are unchanged. Store this objective
as a separate condition, retain the original pilot, and do not compare models
across objectives as if their recipes matched. All models use the same rule.

Balanced-loss follow-up job3171939 completed. All four models remain at about
50% conjunction balanced accuracy and mostly predict YES. Final audit confirms
correct labels, matched data and the intended holdout. Stop at eight pilot
runs, without expanding variants, seeds or record count. See FINDINGS.md.

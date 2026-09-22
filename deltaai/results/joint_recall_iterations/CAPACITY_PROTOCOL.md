# Joint-recall capacity sweep

The 128-binding mixed-size study produced a completed SMat GDN advantage but a
negative default-learning-rate SMat Mamba comparison. Following the user's question
about task difficulty, this separate study extends the same explicit-record task.
It retains the earlier outcomes and measures a full capacity curve rather than
combining winning points from different training datasets.

Fixed training dataset: 36,000 tables at each of 4,16,128,256,512 bindings,
corresponding to C1/K4,C2/K8,C8/K16,C16/K16,C32/K16. Total180,000 examples.
The vocabulary contains32 context IDs,512 key IDs,16 value IDs, and3 special tokens.
The512-key domain supports the paired unique-key control at the largest load and
applies identically to all models and sizes. Every information record contains
context,key,value; queries contain context,key,PAD. Record and query orders are
independently shuffled; every binding is queried. No answer is fed back as input.

Use the unchanged native and SMat d=3 implementations, two layers, width64,
training seed123, BF16, batch256, AdamW weight decay0.1, and32 epochs. The primary
recipe uses LR0.003 with cosine decay by epoch; also run native LR0.01 controls
and compare against the stronger completed native validation setting. GDN retains
its original scalar decay and delta transitions. There are705 updates per epoch,
22,560 in total, and5.76 million example presentations per main run.

The completed earlier mixed-size comparison favored LR0.01 for SMat Mamba
(91.47% macro validation versus approximately84.8% with LR0.003), while GDN
favored LR0.003. Therefore the capacity study also includes SMat Mamba atLR0.01
on the identical cached dataset. This carries forward its stronger completed
development recipe; final selection still uses the capacity validation split,
and native Mamba is evaluated at both learning rates as well.

Before full runs, execute separate200-update, batch32 pilots of all four models.
These are integration checks and early learning measurements, not final comparisons.
Do not transfer their checkpoints into the main runs or select final models using
their scores. The expanded dataset's target decoding, pairing, file hashes, and
split-disjointness audit must pass before scaling up.

Development validation has2,000 tables per cell; development test has4,000 per
cell. Select final-epoch checkpoints using validation only. Report every load,
macro query accuracy, and exact-table accuracy, including negative outcomes.
The original mixed-size and large-only training results remain separate.

If both primary SMat d=3 variants beat the strongest tested native models in
completed macro and largest-cell validation scores and exceed the single-field
control, proceed with the all-d extension below. After that sweep finishes, lock
all eight checkpoint hashes and evaluate 10,000 fresh tables per cell with split
code9001. Use paired whole-table bootstrap
intervals with2,000 draws and equal-cell stratification for macro scores. A confirmed
win requires positive macro and512-binding differences with95% intervals excluding
zero. The largest-cell SMat-minus-single-field-control difference must also have
a positive paired 95% interval. This control chooses, for each entire table, the
better of the key-only and context-only majority predictors. Report both individual
controls as well. This criterion is recorded before any fresh confirmation data.
Check every native candidate's largest-cell score as well as its macro score; if
another native checkpoint is stronger there, include it in confirmation. Report
all cells and any ceiling effects. A single training seed supports only a
seed-conditional result, and these splits do not establish OOD generalization.

No fresh confirmation tables have been evaluated when this protocol is recorded.
If confirmation fails, preserve it and use a new independent split code for any
subsequent selected comparison, following `CONFIRMATION_PROTOCOL.md`.

## All-d extension requested by the user

If the completed d=3 capacity comparison is positive for both backbones against
the strongest native settings, extend this same dataset to d=2 and d=4 for both
backbones. Commands are recorded in `pending_all_d_tasks.json`; the completed
gate in `all_d_gate.json` authorized their addition to the runnable queue. Use
LR0.003, weight decay0.1, width64, two layers,
32 epochs, batch256, and seed123 for the primary d=2,3,4 sweep. The existing d=3
LR0.01 trial remains a separately reported optimization control. Do not suppress
negative variants or change the dataset for an individual d. Report per-load
query and exact-table accuracies and parameter counts for all eight models.
Lock all variants before any fresh confirmation evaluation; use the same fresh
tables for every d. A successful d=3 result does not imply that d=2 or d=4 wins.

For this expanded study, the fresh evaluation locks all eight final checkpoints
after the d=2/4 runs finish. Use the common-LR0.003 checkpoint for every SMat d
and the strongest completed native validation setting for each backbone. Keep
the d=3 LR0.01 run in development results rather than substituting it into the
common-recipe dimension sweep. The primary positive-result criterion remains
d=3 in both backbones; report d=2 and d=4 regardless of their outcomes. Individual
95% intervals for the added dimensions are descriptive, not a basis for selecting
a new winner after seeing the fresh split.

The completed native GDN LR0.01 control is stronger at512 bindings (14.6762%)
but weaker in macro accuracy (48.8159%) than LR0.003 (13.5938% and49.3147%).
Therefore fresh confirmation must include both GDN native checkpoints. The main
common-recipe comparison uses LR0.003; the separately labeled LR0.01 native
control is also evaluated against every SMat d at every load and in macro.
The d=3 confirmation criterion must beat both controls, with positive paired
95% intervals in macro and at512 bindings. This is recorded before fresh data.

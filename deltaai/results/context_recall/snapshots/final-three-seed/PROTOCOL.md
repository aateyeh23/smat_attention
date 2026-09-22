# Context-dependent recall at 500 steps

This is an explicit-context variant of joint recall: every record is a
(context, key, value) triple. Records are globally shuffled and placed in random
three-token slots, with gaps. Four distinct random records are queried as
(QUERY, context, key); answers are never supplied as input. This is not the
published block-context encoding and is not claimed to be an exact replication.

Initial pilot: eight records, two contexts, four keys per context, sequence length
64, width64, two layers, seed123, exactly500 steps. Compare unique keys (reuse0)
with keys shared across both contexts (reuse1). Both conditions use identical
random draws: context IDs, record values/order/positions, and query record indices
are matched. Only the reused key identities change. Context tokens are sampled
from16 IDs, keys from64 IDs, values uniformly from16 possible labels. Key-value
and context-value assignments are fresh for every example, with independent
train/validation/test streams. The input never uses SMat geometry or reset positions.

Reuse the existing matched trainer: batch32, AdamW lr0.01/weight_decay0.1,
cosine decay over500 steps, clipping1, BF16. Validation1024 examples every100
steps; independent final test4096 examples. Existing model implementations are
unchanged: native Mamba2/GDN and current four-read Mamba SMat / selected GDN
transport SMat. The first pilot compares d1 and d4 on both backbones, prioritizing
the strongest candidate from the previous study. Additional variants/seeds are
conditional on these real pilot results. Training uses DeltaAI ghx4-interactive.

Report full-vocabulary per-query accuracy and exact accuracy across all4 queries.
Also report a key-only oracle (knows each key's values but ignores context), a
context-only oracle (ignores key), and a query-blind global-value-mode control.
For shared keys, context_binding_gap_pp measures correct-query accuracy minus
accuracy against the same key's value in another context. A context-ignorant
strategy has expected gap0. It is undefined for the unique-key control, where
there is no same-key alternative context. A separate generic binding gap compares
against all other records. Raw accuracy without context binding is insufficient.

Training code: context_recall.py, reusing synthetic_memory.run via its optional
task API. Existing state/variable generators and recipes are preserved. Current
model sizes are matched within each backbone, but SMat has additional parameters,
memory state, its routing auxiliary loss, and its existing halfway reset. This
is a comparison of the existing implementations, not an isolated mechanism ablation.

Pilot outcome: at8 records all tested models have near-zero context-binding gaps
under full key reuse; raw accuracies are not evidence of contextual retrieval.
Uniform calibration: reduce records8→4, holding two contexts, T64, all other data
and training settings, and500 steps fixed for every model. Repeat both reuse
conditions and the same d1/d4 backbones. The original8-record results are retained.
A small post-training audit independently decoded16 generated examples and
verified that paired reuse conditions differ only in key identities.

After both real d1/d4 pilots, complete the single-seed screen with d2 and d3
at both4 and8 records and both reuse conditions (32 total cells, reusing the
16 pilot cells). This checks whether the earlier choice of d4 missed a better
variant. Use two interactive GPUs, one per backbone, with unchanged recipes.
No higher load or longer training is introduced.

Four-record d1/d4 outcome: native GDN learns context binding (91.02% with shared
keys), whereas current GDN+d4 reaches23.63% and Mamba+d4 is effectively tied with
Mamba around26.5%, without context binding. Thus the calibrated task is learnable
but d4 is not helped by key reuse. The pending two-GPU all-d completion was
estimated to wait22 minutes; replace it with a short single-GPU interactive job
for the identical remaining cells. Completed cells are reused, no recipe changes.

All32 one-seed cells completed and tables/plots were archived under
snapshots/one-seed-before-confirmation before additional seeds. Shared-key load4
has a Mamba+d3 candidate:67.20% vs26.46% for Mamba, context gap47.56pp versus
-0.12pp. The other Mamba variants have no context-binding gain in this seed;
all GDN SMat variants lose to GDN. At8 records every variant has a near-zero
context gap. No high-load capacity claim is supported.

Confirm load4 with seeds123,456,789 across BOTH reuse conditions and ALL eight
models:48 selected cells,16 reused and32 new500-step runs. This includes all
negative/tied variants and the unique-key control. The8-record screen remains
single-seed. No further calibration, optimizer or architecture changes.

Final outcome: all64 runs completed at500 steps. The four-record three-seed
shared-key means are Mamba39.07% vs best-mean SMat d3 35.89%, and GDN89.87%
vs best-mean SMat d3 35.08%. Mamba+d3's seed123 context-binding success did not
repeat in seeds456/789. All shared-key SMat means are below their backbones.
The positive d3 change in relative advantage between unique/shared keys is
exploratory and does not constitute an absolute win. See FINDINGS.md for the
complete interpretation. The final64-run data/metric audit passed.

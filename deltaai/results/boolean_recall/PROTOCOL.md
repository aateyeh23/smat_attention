# Two-key bit retrieval and conjunction: pilot protocol

Three matched conditions:

1. `retrieval`: query two distinct keys and predict their two stored bits.
2. `direct_and`: supply those two bits directly and predict their AND.
3. `retrieval_and`: query the same two keys and predict the AND of their bits.

Memory rows are `(key, bit)`, with unique keys sampled from16 IDs. Rows are
shuffled into random two-token slots with gaps, independently of model geometry.
The query suffix is `QUERY keyA keyB ANSWER_A ANSWER_B`. Retrieval predicts bitA
at ANSWER_A and bitB at ANSWER_B. The AND conditions predict only at ANSWER_B;
direct_and substitutes the two supplied bits for the two key tokens. The
answer-marker tokens are fixed markers, never teacher-forced answers.

All conditions consume identical random draws and have identical memory rows,
record positions/order and queried bit pairs. The direct control retains the
same irrelevant memory as the retrieval conditions. Within each generated
batch, the four query truth cases00/01/10/11 are equally frequent, in shuffled
example order. Query keys are distinct and sampled uniformly; their stored
bits are set to the chosen case, while other bits are independent fair bits.
This stratifies the truth table without selecting record positions or routes.

Pilot: four records, T64, width64, two layers, seed123, native Mamba-2/GDN and
their existing SMat d3 variants:12 runs. Exactly500 updates, batch32, AdamW
lr.01/wd.1, cosine to zero, clipping1, BF16, ordinary unweighted cross-entropy.
Use the established trainer and model implementations unchanged. Retrieval
supervises two labels/example and the AND tasks one, with mean loss in each
case; equal optimizer steps do not mean equal supervised-token counts.
Validation1024 examples every100 steps, final independent test4096 examples
(1024 per truth case). Fresh independent training data at each step.
Launch the real pilot before extended checks: job3172006, one GPU on DeltaAI
ghx4-interactive. Additional seeds/loads are conditional on pilot learning.

Report full-vocabulary accuracy, exact accuracy, all four truth-case accuracies
and exact accuracies, and the worst truth-case exact accuracy. An always-zero
AND output scores75% overall and0% on11; overall accuracy alone is insufficient.
Also report AND balanced accuracy (equal weight on output0 and output1) and a
query-blind count oracle using the memory's bit counts. For retrieval, compute
AND deterministically from the two predicted bits as a diagnostic; invalid
vocabulary predictions count as errors. This does not retrain the model.

The binding diagnostic compares predictions with other keys' bits (retrieval)
or other unordered pairs' AND outcomes (AND tasks), excluding the queried
key/pair. It measures specificity, not a causal routing contribution. For
direct_and, successful specificity can come entirely from the supplied bits.

These are existing-implementation comparisons: SMat adds memory, parameters,
routing loss and its local-state reset. No parameter matching or mask-only
ablation is claimed. Three task conditions localize failures but cannot by
themselves distinguish representation limits from optimization. Matching RNG
seeds/data does not guarantee bitwise deterministic custom-kernel training.

Pilot outcome: all four direct-AND models reached100% on every truth case.
Native GDN retrieved both bits correctly94.41% of the time. End-to-end AND
accuracies were GDN91.58%, GDN+SMat d3 84.45%, Mamba85.99%, Mamba+SMat d3
74.95%; the latter had0% accuracy on11. The dataset/metric audit passed.
Preserve the complete12-run screen under snapshots/one-seed-before-confirmation.
Confirm all three conditions and all four models with seeds123/456/789, reusing
the first12 cells and adding24 runs (36 total). No changes to data, loss,
architecture or500-step schedule. Confirmation job3172032.

Evaluation-only additional shortcut control: an oracle given the total number
of1 bits and only one queried bit can predict AND above the count-only oracle.
For a known0 it predicts0; for a known1 it predicts whether a majority of the
remaining records have1. With four IID fair-bit records its expected accuracy
is87.5%, versus81.25% for counts only. Compute the finite-test references for
both first-bit-known and second-bit-known cases. These are privileged diagnostic
references, not trained baselines or universal bounds. They do not change
training, data, or the pre-existing truth-case metrics.

Final outcome: all36 runs completed at500 steps and passed the data/metric
audit. All12 supplied-bit AND runs reached100%. Mean two-bit retrieval exact
accuracy was Mamba59.81%, Mamba+SMat46.20%, GDN87.00%, GDN+SMat67.55%.
Mean end-to-end AND accuracy was83.54%,78.88%,88.53%,87.81%, respectively.
GDN+SMat won seed789 and lost the other two; Mamba+SMat lost all three. No
consistent SMat advantage was established. Stop at the planned36-run grid;
preserve all results and report the component controls and shortcut references.

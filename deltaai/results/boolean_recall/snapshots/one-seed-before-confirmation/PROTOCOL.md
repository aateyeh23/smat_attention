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

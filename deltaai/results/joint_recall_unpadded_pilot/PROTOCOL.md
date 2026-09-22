# Block-context joint recall: fixed dataset, full epoch budget

This study follows the multi-query joint-recall task structure of Zhan et al.,
*Overcoming Long-Context Limitations of State-Space Models via Context-Dependent
Sparse Attention* (2025), https://arxiv.org/abs/2507.00449 . The official generator
was inspected at HAX commit73d3c18b5dfd554392d6e278ff2f2564629cab97:
https://github.com/DeepGraphLearning/HAX/blob/73d3c18b5dfd554392d6e278ff2f2564629cab97/scripts/create_dataset.py .

Each information block starts with one context token, then shuffled key/value
pairs. The same key set is reused across contexts, with independent uniformly
sampled categorical values. Inquiry blocks permute context and key order and
query every table entry. BOS starts the information half and SEP starts the
inquiry half. Predict at each inquiry key. All inquiry answer slots contain
PAD, never the true answers. Contexts and keys use distinct token domains.

Documented adaptations: the author's implementation supplies previous inquiry
answers as input; ours masks them, following our MQAR convention. We use five
table sizes, randomly sampled context/key token IDs,180k training examples and
the32-epoch MQAR budget. The author's published dataset has contexts/keys5–16
and about1.4M training examples. This is a block-context joint-recall study,
not an exact reproduction of the HAX dataset, training or reported numbers.

## Dataset and controls

| Contexts | Keys/context | Entries | Sequence length | Training examples | Validation | Test |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 4 | 22 | 100,000 | 1,000 | 2,000 |
| 2 | 4 | 8 | 38 | 20,000 | 1,000 | 2,000 |
| 4 | 4 | 16 | 74 | 20,000 | 1,000 | 2,000 |
| 4 | 8 | 32 | 138 | 20,000 | 1,000 | 2,000 |
| 8 | 8 | 64 | 274 | 20,000 | 1,000 | 2,000 |

Each training condition contains exactly180,000 fixed examples; validation5,000
and test10,000. All conditions and model seeds use data seed20260918, with
independent train/validation/test streams. The unique-key control changes only
key identities so every key is globally unique. Contexts, values, block/order
permutations, query positions/targets and sequence lengths match the shared-key
condition exactly. Domains:32 context IDs,128 key IDs,16 answer values.
No learned model or SMat geometry is consulted by the data generator.

Sequences use the exact even length2+2C+4CK, without geometry-dependent padding
or gaps. The equal-length information and inquiry blocks naturally place their
boundary at the midpoint, where the existing reset SMat models reset their
local branch. This structural alignment is disclosed and is not an independent
test of arbitrary boundary locations. All length-specific routing parameters
are prebuilt before optimizer construction.

## Training and evaluation

Models: current native Mamba-2/GDN and their existing SMat d3 variants, width64,
two layers, head/state dimension16. Both shared/unique conditions and seeds
123/456/789:24 complete training runs. Models keep their existing architecture,
extra SMat parameters/memory, auxiliary loss and reset behavior.

Batch256; each epoch shuffles examples within each cell and interleaves the707
resulting batches. Every sample is used once per epoch, including partial final
batches. Exactly32 epochs:22,624 optimizer updates and5,760,000 example
presentations per model/condition/seed. AdamW lr.01, weight decay.1, cosine decay
stepped once per epoch. No early stopping. BF16 autocast and the established
custom kernels are used; gradient clipping is disabled as in the MQAR trainer.
This matches MQAR's data exposure and epoch/optimizer schedule, not the HAX
paper's separate training recipe. Training is not bitwise deterministic.

Evaluate validation after each epoch, preserving best-validation and final
checkpoints. Evaluate both on the separate test only after all32 epochs. Final
epoch test accuracy is the primary fixed-budget endpoint; best-validation
checkpoint test accuracy is secondary. Report each table size, macro accuracy
across the five cells, all-query exact accuracy, training curves, parameter
counts, and seed variation. Test results are not used to choose checkpoints.

## Execution

All GPU training uses DeltaAI ghx4-interactive. Pilot job3172131 trains64 actual
updates for each of the four shared-key seed123 models using the full cached
dataset and32-epoch schedule. These checkpoints resume into the full runs;
64 updates are an implementation check, not the performance endpoint. Full
training proceeds after verifying finite training and the five sequence shapes.
Checkpoints retain intra-epoch cursor, optimizer/scheduler/RNG state, and model
routing step counters. Validation-best checkpoints also retain routing counters.

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
| 2 | 2 | 4 | 64 | 100,000 | 1,000 | 2,000 |
| 2 | 4 | 8 | 64 | 20,000 | 1,000 | 2,000 |
| 4 | 4 | 16 | 76 | 20,000 | 1,000 | 2,000 |
| 4 | 8 | 32 | 140 | 20,000 | 1,000 | 2,000 |
| 8 | 8 | 64 | 276 | 20,000 | 1,000 | 2,000 |

Each training condition contains exactly180,000 fixed examples; validation5,000
and test10,000. All conditions and model seeds use data seed20260918, with
independent train/validation/test streams. The unique-key control changes only
key identities so every key is globally unique. Contexts, values, block/order
permutations, query positions/targets and sequence lengths match the shared-key
condition exactly. Domains:32 context IDs,128 key IDs,16 answer values.
No learned model or SMat geometry is consulted by the data generator.

Unpadded length is2+2C+4CK. The first real pilot exposed that the existing SMat
kernel/mask constructor does not support all such lengths (the shortest length22
produced no valid mask). Use length max(64,4*ceil(raw_length/4)) and pad the end
of each component equally, for every model and both conditions. This preserves
the published equal-component boundary at the midpoint, where the existing
reset SMat models reset their local branch. This structural alignment is
disclosed and is not an independent test of arbitrary boundary locations.
All length-specific routing parameters are prebuilt before optimizer construction.

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

Job3172131 failed at SMat construction after the two native models ran64 steps.
Its unpadded files and checkpoints are retained under joint_recall_unpadded_pilot.
Those checkpoints are not reused after the common padding compatibility fix;
restart all four pilot models on the padded datasets before the full campaign.

Padded pilot job3172150 completed all four models successfully. Full training
uses the frozen Python files under source/, with explicit output root; pilot
checkpoints at update64 are resumed. The regenerated dataset passed decoding,
pairing, split-disjointness and SHA256 checks (data_audit.json).

Full campaign job3172175 uses four GPUs on ghx4-interactive. Continuation
job3172209 is queued after successful completion of that job. If the two-hour
allocation ends before the complete grid is done, successful continuation
jobs resume checkpoints and queue the next slice on the same partition.
Training failures stop the chain. There is a review limit of12 continuation
slices; completed runs are skipped and never repeated.

The analytical test controls include key-only majority, last-value-for-key and
context-only majority lookup. Their input includes the full information table,
so these are context/key ablations of an oracle, not learned model baselines.
Saved test predictions are also scored on queries where the target differs
from the most recent information value for that key. This separates retrieval
of older context-specific values from a last-write lookup. These diagnostics do
not affect training, stopping, checkpoint selection or dataset generation.

All train/validation/test splits sample the same token domains and table-size
distribution independently. This tests in-distribution joint recall; it is not
a held-out-combination or out-of-distribution generalization claim.

Pending continuation3172209 was replaced by3172219 before it ran, to ensure
reports and result audits refresh even if the first allocation finishes the
entire grid. The original pending job was cancelled; training was unaffected.

User changed the objective to iterative development until both SMat variants
beat their baselines, using one seed. The original three-seed job3172175 and
pending continuation3172219 were cancelled. Resume only seed123 from epoch
checkpoints; completed runs are reused. Further changes are development trials
selected on validation, with matched baselines and a fresh final test for the
selected configuration. All earlier results remain available.

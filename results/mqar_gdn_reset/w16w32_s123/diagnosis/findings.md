# Early GDN+SMAT/reset MQAR drop

Read-only investigation of width 16 and 32, d=3, seed 123, epoch-4 checkpoints.

- Full validation reproduces the saved accuracies exactly (24.1596875% / 36.90125%). This is a real model change, not a checkpoint evaluation mismatch.
- Rounded logged epoch accuracies: w16 0.108%, 30.4%, 16.6%, 24.2%; w32 11.9%, 54.3%, 27.4%, 36.9%. Both regress in epoch 3 and partially recover in epoch 4.
- Routing anneals over 1000 training forwards, while there are 707 batches/epoch. It is already fully hard by epoch-2 validation; no soft-to-hard transition occurs between epoch-2 and epoch-3 validation.
- The existing Zoology loader uses shuffle=False and concatenates batches by task cell. Each epoch is 391 batches of (length64,4 pairs), then 79 each of (128,8), (256,16), (256,32), (256,64). This is a repeated fixed curriculum, not randomly interleaved mixture batches. Interference/forgetting is a plausible mechanism, not yet established by a shuffled training comparison.
- Width32 mean logged loss over the final 20 batches of the 64-pair block rises from 5.461 in epoch2 to 8.354 in epoch3, confirming degradation also on training batches. These approximate values are parsed from rounded tqdm logs and include the auxiliary loss.
- Inference-only soft-hash intervention: w16 24.16 -> 24.23%; w32 36.90 -> 44.93%. Routing sensitivity explains some recoverable performance at width32, not the full earlier decline. This is not evidence that training with soft routing will yield the same improvement.
- Disabling G while keeping the reset: w16 13.53%, w32 20.75%. Both models still rely on G.
- Earlier epoch checkpoints were overwritten; there is no direct parameter/assignment comparison to the peak epoch. No training configuration or checkpoints were modified by these probes.

Next discriminating experiment: interleave batches across task cells with matched seed/model/LR, retaining the existing baseline recipe as a control. Lower LR (current 0.01) is a separate stability ablation, not a proven fix.

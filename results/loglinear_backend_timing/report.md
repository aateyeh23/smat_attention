# Optimized Log-Linear benchmark on the GPU cluster

Measured full two-layer training steps, width 64, batch 256, BF16 autocast, AdamW, actual joint-recall training examples. Three warmup steps followed by five timed steps per cell. Timings include forward, backward and optimizer update; compilation, validation, checkpointing and scheduler waits are excluded. GPU: NVIDIA GH200 120GB. Each backend ran on an exclusively allocated GPU; nodes may differ.

| Backbone | Length | Dense seconds/step | Optimized seconds/step | Speedup |
|---|---:|---:|---:|---:|
| mamba2 | 64 | 0.0098 | 0.0093 | 1.06x |
| mamba2 | 100 | 0.0103 | 0.0103 | 1.00x |
| mamba2 | 772 | 0.2830 | 0.0280 | 10.09x |
| mamba2 | 1540 | 1.1468 | 0.0548 | 20.93x |
| mamba2 | 3076 | 5.1244 | 0.1108 | 46.26x |
| mamba2 | Equal-cell mean | 1.3149 | 0.0427 | 30.83x |
| gdn | 64 | 0.0171 | 0.0280 | 0.61x |
| gdn | 100 | 0.0174 | 0.0303 | 0.57x |
| gdn | 772 | 0.2555 | 0.2021 | 1.26x |
| gdn | 1540 | 1.1889 | 0.2298 | 5.17x |
| gdn | 3076 | 6.2994 | 0.3892 | 16.18x |
| gdn | Equal-cell mean | 1.5556 | 0.1759 | 8.84x |

Mamba uses an isolated upstream WEAK Triton port with block size 16, four warps and two stages, plus matching q/k/v dtypes. Value inputs promoted to FP32 by the Mamba dt multiplication are cast to BF16 inside this kernel; output is restored to the input value dtype. Causal zero padding makes lengths multiples of 64. GDN uses a separately implemented FP32 binary-tree evaluation, batch chunks of 256 for short lengths and 64 for long lengths, with activation checkpointing. The upstream GDN kernel was rejected after key/beta gradient discrepancies of about 32%/47% at length 64.

Both reported paths passed sampled forward/backward checks against the dense reference across lengths 64, 100, 256, 772, 1540 and 3076. Full-batch training-step gradients were finite. Trained seed-123 checkpoint logits and concatenated parameter gradients were also compared on four actual examples per cell with matched dropout RNG. These checks establish sampled numerical agreement, not identical training trajectories or final accuracy.

- mamba2: maximum checkpoint logit relative L2 difference 0.619%; maximum concatenated parameter-gradient difference 2.620%; checkpoint at step 15510.
- gdn: maximum checkpoint logit relative L2 difference 0.053%; maximum concatenated parameter-gradient difference 0.559%; checkpoint at step 10575.

All production five-seed runs remain on their original dense backend. No benchmark writes to their checkpoints, recipes, results or frozen sources. This is a throughput experiment, not a new accuracy result. Short GDN sequences are slower with the tree implementation; the aggregate gain comes from the longer cells.

Measurement provenance: job 3182567 completed all dense Mamba measurements, then failed during the subsequent optimized-kernel dtype check; that mixed-dtype issue was repaired. Jobs 3182587 (optimized Mamba and dense GDN), 3182572 (tree64 both backbones), and 3182588 (trained-checkpoint comparisons) completed successfully. Initial 3182561 was intentionally canceled to remove benchmark-only gradient clipping; its outputs were not used.

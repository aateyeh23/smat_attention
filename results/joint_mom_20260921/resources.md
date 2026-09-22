# Validated resources

Counts are for the complete two-layer model. Matrix bytes provision two heads per layer at length 3076; convolution histories are additional. Peak CUDA allocation is measured during batch256 training.

| Variant | Trainable parameters | Matrix state bytes/example | Conv cache bytes if FP32 | Peak CUDA allocated GiB | Warm max-length step seconds |
|---|---:|---:|---:|---:|---:|
| mom_profiles | 1579368 | 692224 | 519168 | 4.80 | 1.009 |
| mom_bytes | 3228264 | 1437696 | 1078272 | 7.35 | 2.940 |

Independent FP32 recurrence relative forward error: 0.006818965543061495
Relative input/expert-weight gradient errors: [0.008295820094645023, 0.00805249810218811]

All five lengths passed finite loss and nonzero router/expert-gradient checks. Future-token causality, batch isolation, and train/eval agreement passed.

Validated GPU optimization results

H100, 16 layers, width 1024, 16K sequence length. Matched pairs use the same GPU, seed, data windows, batch=2, five warmup and twenty timed optimizer steps. Includes forward, backward and AdamW.

| Model | Reference tokens/s | Optimized tokens/s | Speedup | Reference GB | Optimized GB |
|---|---:|---:|---:|---:|---:|
| GDN | 91,298 | 102,765 | 1.126× | 35.28 | 30.85 |
| GDN + SMAT d=2 | 55,968 | 61,966 | 1.107× | 49.40 | 43.22 |
| GDN + SMAT d=3 | 52,070 | 58,784 | 1.129× | 50.73 | 44.38 |
| GDN + SMAT d=4 | 43,267 | 48,820 | 1.128× | 53.69 | 46.71 |

Optimized: fused Triton top-4 bias/selection/softmax and grouped reads, existing FLA fused RMSNorm, existing FLA recurrence/delta writes; cuBLAS retained for projection/aggregation. Baseline GDN benefits from RMSNorm only.

D=4 at batch 3: 52,626 tokens/s, 66.95 GB peak; 10.56 hours per 2B tokens before evaluation, checkpoint and compilation overhead. Final training uses microbatches 3+3+2 for a shared 8-sequence effective batch.

Validation: standalone output/gradient references, full d=2/3/4 mixer parameter gradients, RMSNorm gradients, fixed-window causal generation, exact model/optimizer checkpoint restoration. Resumed two-step BF16 training varies about as much as independent uninterrupted repetitions (largest parameter difference 0.0003323 vs 0.0003320); not bitwise deterministic.

Sparse aggregation and custom dense tensor-core aggregation/projection were benchmarked and rejected as slower. V1 full-mixer checks accidentally loaded the frozen reference module and are excluded; v3 verified the corrected import and kernels. V4 reports the selected production path.

MQAR width sweeps — seed 123, 32 epochs; learned 1-write/4-read SMAT.

GDN has 2 heads. Mamba2 uses its native 2× expansion: 2/4/8 heads at width 16/32/64. Head and state dimensions are 16.

| Family | Width | Variant | Epoch | Accuracy | 64-pair accuracy | Loss |
|---|---:|---|---:|---:|---:|---:|
| gdn | 32 | Baseline | 32/32 | 69.86% | 10.22% | 1.64 |
| gdn | 32 | SMAT d=2 | 32/32 | 59.79% | 10.70% | 3.65 |
| gdn | 32 | SMAT d=3 | 32/32 | 43.57% | 5.14% | 6.31 |
| gdn | 32 | SMAT d=4 | 32/32 | 68.07% | 15.99% | 2.10 |
| gdn | 64 | Baseline | 32/32 | 43.38% | 1.16% | 3.53 |
| gdn | 64 | SMAT d=2 | 32/32 | 43.96% | 7.82% | 5.19 |
| gdn | 64 | SMAT d=3 | 32/32 | 40.50% | 5.30% | 6.33 |
| gdn | 64 | SMAT d=4 | 32/32 | 0.03% | 0.03% | 8.72 |
| mamba2 | 16 | Baseline | 32/32 | 42.03% | 3.35% | 3.64 |
| mamba2 | 16 | SMAT d=2 | 32/32 | 43.12% | 3.42% | 3.86 |
| mamba2 | 16 | SMAT d=3 | 32/32 | 59.47% | 28.20% | 2.95 |
| mamba2 | 16 | SMAT d=4 | 32/32 | 44.67% | 3.94% | 4.05 |
| mamba2 | 32 | Baseline | 32/32 | 77.24% | 25.99% | 1.19 |
| mamba2 | 32 | SMAT d=2 | 32/32 | 75.68% | 39.60% | 1.31 |
| mamba2 | 32 | SMAT d=3 | 32/32 | 83.99% | 68.11% | 1.09 |
| mamba2 | 32 | SMAT d=4 | 32/32 | 84.22% | 56.33% | 1.04 |
| mamba2 | 64 | Baseline | 32/32 | 83.92% | 44.38% | 0.77 |
| mamba2 | 64 | SMAT d=2 | 32/32 | 90.51% | 80.73% | 0.72 |
| mamba2 | 64 | SMAT d=3 | 32/32 | 90.10% | 74.43% | 0.74 |
| mamba2 | 64 | SMAT d=4 | 32/32 | 93.02% | 85.41% | 0.61 |

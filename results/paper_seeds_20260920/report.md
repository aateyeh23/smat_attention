# Paper seed campaign: 154/162 new runs complete

Final epoch only. Values are percent accuracy; SD is the sample standard deviation across training seeds.
MQAR data and whole-batch ordering remain fixed; joint-recall data remain fixed and batch ordering follows the original seed-dependent trainer.
Table 3 uses equal-cell validation accuracy. Independent final-test results are retained separately.

| Task | Backbone | Width | Variant | Seeds completed | Mean | SD |
|---|---|---:|---|---:|---:|---:|
| joint | gdn_current | 64 | Native | 5/5 | 54.63 | 5.22 |
| joint | gdn_current | 64 | SMat d=2 | 5/5 | 49.05 | 4.76 |
| joint | gdn_current | 64 | SMat d=3 | 5/5 | 56.86 | 4.59 |
| joint | gdn_current | 64 | SMat d=4 | 5/5 | 60.35 | 7.10 |
| joint | mamba2 | 64 | Native | 5/5 | 60.23 | 18.00 |
| joint | mamba2 | 64 | SMat d=2 | 5/5 | 58.24 | 6.33 |
| joint | mamba2 | 64 | SMat d=3 | 5/5 | 60.20 | 3.25 |
| joint | mamba2 | 64 | SMat d=4 | 5/5 | 62.43 | 13.46 |
| joint_loglinear | mamba2 | 64 | Log-Linear | 2/5 | 51.30 | 10.55 |
| mqar | gdn | 16 | Log-Linear | 5/5 | 47.20 | 10.35 |
| mqar | gdn | 16 | Native | 5/5 | 47.26 | 7.70 |
| mqar | gdn | 16 | SMat d=2 | 5/5 | 48.92 | 4.87 |
| mqar | gdn | 16 | SMat d=3 | 5/5 | 44.43 | 4.09 |
| mqar | gdn | 16 | SMat d=4 | 5/5 | 43.02 | 3.22 |
| mqar | gdn | 32 | Log-Linear | 5/5 | 66.11 | 7.87 |
| mqar | gdn | 32 | Native | 5/5 | 65.39 | 7.47 |
| mqar | gdn | 32 | SMat d=2 | 5/5 | 67.06 | 5.16 |
| mqar | gdn | 32 | SMat d=3 | 5/5 | 75.66 | 6.08 |
| mqar | gdn | 32 | SMat d=4 | 5/5 | 77.80 | 8.07 |
| mqar | gdn | 64 | Log-Linear | 5/5 | 78.12 | 2.41 |
| mqar | gdn | 64 | Native | 5/5 | 72.42 | 7.42 |
| mqar | gdn | 64 | SMat d=2 | 5/5 | 71.26 | 12.59 |
| mqar | gdn | 64 | SMat d=3 | 5/5 | 82.65 | 7.61 |
| mqar | gdn | 64 | SMat d=4 | 5/5 | 79.08 | 6.54 |
| mqar | mamba2 | 16 | Log-Linear | 5/5 | 44.57 | 7.18 |
| mqar | mamba2 | 16 | Native | 5/5 | 41.28 | 2.67 |
| mqar | mamba2 | 16 | SMat d=2 | 5/5 | 48.66 | 4.18 |
| mqar | mamba2 | 16 | SMat d=3 | 5/5 | 58.03 | 7.45 |
| mqar | mamba2 | 16 | SMat d=4 | 5/5 | 51.08 | 7.46 |
| mqar | mamba2 | 32 | Log-Linear | 5/5 | 72.66 | 7.02 |
| mqar | mamba2 | 32 | Native | 5/5 | 73.17 | 6.58 |
| mqar | mamba2 | 32 | SMat d=2 | 5/5 | 74.79 | 2.35 |
| mqar | mamba2 | 32 | SMat d=3 | 5/5 | 77.22 | 5.27 |
| mqar | mamba2 | 32 | SMat d=4 | 5/5 | 80.86 | 4.41 |
| mqar | mamba2 | 64 | Log-Linear | 5/5 | 86.56 | 2.00 |
| mqar | mamba2 | 64 | Native | 5/5 | 86.60 | 4.24 |
| mqar | mamba2 | 64 | SMat d=2 | 5/5 | 91.92 | 0.96 |
| mqar | mamba2 | 64 | SMat d=3 | 5/5 | 92.74 | 2.58 |
| mqar | mamba2 | 64 | SMat d=4 | 5/5 | 93.26 | 1.37 |

The draft’s GDN width-64 d=2 entry is 89.71; the saved seed-123 final metric is 89.6834375. Aggregation uses the saved metric.
Original Table 2 Log-Linear results are verified from the second cluster commit 3265a5df. Unfinished runs are never counted as final results.

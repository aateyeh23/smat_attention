# Fast joint-recall Log-Linear results

10/10 runs complete.
Final epoch 32. Accuracy in percent. Validation is the equal-cell metric used in Table 3; final-test accuracy is separate. These are fresh optimized-backend runs, not resumed dense runs.

| Backbone | Seed | Final validation | Final test | Status |
|---|---:|---:|---:|---|
| mamba2 | 123 | 84.92 | 84.97 | Complete |
| gdn | 123 | 7.92 | 8.00 | Complete |
| mamba2 | 1 | 45.34 | 45.35 | Complete |
| gdn | 1 | 56.32 | 56.39 | Complete |
| mamba2 | 2 | 79.55 | 79.54 | Complete |
| gdn | 2 | 56.37 | 56.41 | Complete |
| mamba2 | 3 | 67.76 | 67.66 | Complete |
| gdn | 3 | 63.73 | 63.77 | Complete |
| mamba2 | 4 | 50.76 | 50.80 | Complete |
| gdn | 4 | 53.30 | 53.28 | Complete |

mamba2: five-seed validation mean 65.67 ± 17.34 (sample SD).

gdn: five-seed validation mean 47.53 ± 22.47 (sample SD).

GDN seed 123 diagnostic: the same final fast checkpoint scores 7.913% with dense evaluation and 7.916% with fast evaluation. The low score persists under both evaluators. The cause of the training divergence from the independent dense run is unresolved; the seed remains in the aggregate.

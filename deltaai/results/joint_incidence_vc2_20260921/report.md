# Degree-matched VC-2 incidence control

Exact VC=2 applies to individual summary supports; four-read unions can have higher VC. Matrices come from a constrained random walk initialized from relabeled geometry, not uniform sampling.

| Run | Epoch | Complete | Latest validation macro (%) | Final test: 128 bindings (%) | Final test macro (%) |
|---|---:|---|---:|---:|---:|
| random_vc2-t17-s123 | 32 | True | 47.53 | 19.24 | 47.47 |
| random_vc2-t17-s456 | 32 | True | 62.69 | 41.52 | 62.75 |
| random_vc2-t17-s789 | 32 | True | 64.13 | 63.30 | 64.13 |

## Paired final-test comparison

| Seed | Variant | 128 bindings (%) | Macro (%) |
|---:|---|---:|---:|
| 123 | geometry | 37.22 | 54.05 |
| 123 | random VC3, draw 17 | 53.53 | 57.92 |
| 123 | random VC2 | 19.24 | 47.47 |
| 456 | geometry | 33.14 | 61.40 |
| 456 | random VC3, draw 17 | 31.62 | 51.57 |
| 456 | random VC2 | 41.52 | 62.75 |
| 789 | geometry | 60.24 | 61.32 |
| 789 | random VC3, draw 17 | 30.78 | 51.43 |
| 789 | random VC2 | 63.30 | 64.13 |

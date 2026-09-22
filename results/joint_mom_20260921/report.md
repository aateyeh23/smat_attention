# Memory-matched MoM top4 comparison

Four routed reads and writes, no shared memory; profile-count or matrix-byte matching. Parameters and total cache are not matched.

| Run | Epoch | Complete | Latest validation macro (%) | Final test: 128 bindings (%) | Final test macro (%) |
|---|---:|---|---:|---:|---:|
| mom_profiles-s123 | 32 | True | 45.38 | 21.64 | 45.29 |
| mom_profiles-s456 | 32 | True | 43.46 | 17.06 | 43.54 |
| mom_profiles-s789 | 32 | True | 42.34 | 17.03 | 42.35 |
| mom_bytes-s123 | 32 | True | 33.62 | 11.00 | 33.70 |
| mom_bytes-s456 | 32 | True | 32.16 | 13.56 | 32.18 |
| mom_bytes-s789 | 32 | True | 39.28 | 13.75 | 39.43 |

## Paired final-test comparison

| Seed | Variant | 128 bindings (%) | Macro (%) |
|---:|---|---:|---:|
| 123 | geometry | 37.22 | 54.05 |
| 123 | MoM profiles | 21.64 | 45.29 |
| 123 | MoM bytes | 11.00 | 33.70 |
| 456 | geometry | 33.14 | 61.40 |
| 456 | MoM profiles | 17.06 | 43.54 |
| 456 | MoM bytes | 13.56 | 32.18 |
| 789 | geometry | 60.24 | 61.32 |
| 789 | MoM profiles | 17.03 | 42.35 |
| 789 | MoM bytes | 13.75 | 39.43 |

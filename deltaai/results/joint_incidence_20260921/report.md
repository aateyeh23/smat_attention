# Joint-recall incidence ablation

GDN, d=3, width 64, two layers; 32 epochs, LR 0.003. Three training seeds. Random incidence uses two graph seeds crossed with every training seed.

Geometry versus random changes only incidence. Independent-bucket controls change the reader; the byte-matched control also changes hash bin counts. Matrix-table bytes are not an implemented autoregressive cache.

| Run | Epoch | Status | Validation macro (%) | Final test macro (%) |
|---|---:|---|---:|---:|
| geometry-s123 | 32 | complete | 53.96 | 54.05 |
| random-t17-s123 | 32 | complete | 57.92 | 57.92 |
| buckets_profiles-s123 | 32 | complete | 44.10 | 44.00 |
| buckets_bytes-s123 | 32 | complete | 36.94 | 37.12 |
| random-t29-s123 | 32 | complete | 49.59 | 49.66 |
| geometry-s456 | 32 | complete | 61.43 | 61.40 |
| random-t17-s456 | 32 | complete | 51.55 | 51.57 |
| buckets_profiles-s456 | 32 | complete | 45.61 | 45.58 |
| buckets_bytes-s456 | 32 | complete | 37.87 | 37.81 |
| random-t29-s456 | 32 | complete | 63.20 | 63.21 |
| geometry-s789 | 32 | complete | 61.32 | 61.32 |
| random-t17-s789 | 32 | complete | 51.44 | 51.43 |
| buckets_profiles-s789 | 32 | complete | 44.87 | 44.88 |
| buckets_bytes-s789 | 32 | complete | 36.65 | 36.71 |
| random-t29-s789 | 32 | complete | 56.27 | 56.25 |

## Final test accuracy by load

Only complete runs appear below.

| Run | 4 bindings | 16 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| geometry-s123 | 99.03 | 89.62 | 37.22 | 27.64 | 16.73 |
| random-t17-s123 | 99.98 | 92.23 | 53.53 | 23.89 | 19.97 |
| buckets_profiles-s123 | 99.36 | 83.28 | 13.20 | 10.61 | 13.56 |
| buckets_bytes-s123 | 99.38 | 60.72 | 12.41 | 6.75 | 6.32 |
| random-t29-s123 | 99.88 | 87.48 | 28.03 | 19.59 | 13.32 |
| geometry-s456 | 99.97 | 95.63 | 33.14 | 42.88 | 35.40 |
| random-t17-s456 | 99.88 | 87.54 | 31.62 | 22.34 | 16.45 |
| buckets_profiles-s456 | 99.94 | 94.08 | 11.32 | 14.30 | 8.28 |
| buckets_bytes-s456 | 99.71 | 51.61 | 21.86 | 8.48 | 7.37 |
| random-t29-s456 | 99.87 | 98.78 | 68.80 | 25.40 | 23.18 |
| geometry-s789 | 99.97 | 96.82 | 60.24 | 31.39 | 18.20 |
| random-t17-s789 | 100.00 | 88.92 | 30.78 | 21.52 | 15.92 |
| buckets_profiles-s789 | 99.71 | 73.14 | 21.94 | 19.16 | 10.46 |
| buckets_bytes-s789 | 99.59 | 50.74 | 16.43 | 10.28 | 6.51 |
| random-t29-s789 | 99.96 | 97.15 | 35.32 | 33.25 | 15.57 |

## Geometry minus random incidence

Random topology draws are averaged within each training seed. Reported variability is across the three paired training-seed differences.

| Cell | Mean difference (pp) | SD across training seeds (pp) |
|---|---:|---:|
| c1-k4 | -0.27 | 0.55 |
| c2-k8 | +2.01 | 2.05 |
| c8-k16 | +2.18 | 22.69 |
| c16-k16 | +9.63 | 8.17 |
| c32-k16 | +6.04 | 8.35 |

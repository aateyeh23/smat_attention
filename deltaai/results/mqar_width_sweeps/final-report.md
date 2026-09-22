Final MQAR results — all 20 runs completed 32 epochs.

Each cell reports **overall accuracy / 64-pair accuracy (%)**. Same seed (123), two layers, learning rate 0.01, and cached MQAR data. GDN uses two heads at widths 32/64; Mamba2 uses 2/4/8 heads at widths 16/32/64. Head/state dimensions are 16. SMAT uses one learned write and four distinct weighted reads per head.

| Backbone | Width | Baseline | SMAT d=2 | SMAT d=3 | SMAT d=4 |
|---|---:|---:|---:|---:|---:|
| GDN | 32 | 69.86 / 10.22 | 59.79 / 10.70 | 43.57 / 5.14 | 68.07 / 15.99 |
| GDN | 64 | 43.38 / 1.16 | 43.96 / 7.82 | 40.50 / 5.30 | 0.03 / 0.03 |
| Mamba2 | 16 | 42.03 / 3.35 | 43.12 / 3.42 | 59.47 / 28.20 | 44.67 / 3.94 |
| Mamba2 | 32 | 77.24 / 25.99 | 75.68 / 39.60 | 83.99 / 68.11 | 84.22 / 56.33 |
| Mamba2 | 64 | 83.92 / 44.38 | 90.51 / 80.73 | 90.10 / 74.43 | 93.02 / 85.41 |

Mamba2 + SMAT shows substantial recall gains, especially at width 64. Its d=4 arm finishes at 93.02% overall and 85.41% on 64-pair recall, compared with 83.92% and 44.38% for the baseline. The best SMAT dimension depends on width and metric; these results do not establish a universal monotonic improvement with d.

GDN + SMAT is not consistently better under this shared recipe. In particular, width-64 d=4 collapsed during training and finished near chance. All final results are retained, including this failure; numbers are final-epoch rather than best-checkpoint values. This is one seed and one shared learning rate, not a tuned or multi-seed comparison.

All 20 final checkpoints, 32-epoch histories, CSV/JSON metrics and source hashes are saved locally and on Modal volume `smat-mqar-width-sweeps`. Both MQAR apps have completed.

[Accuracy curves](accuracy_curves.png) · [PDF](accuracy_curves.pdf) · [LaTeX table](table.tex) · [CSV](summary.csv)

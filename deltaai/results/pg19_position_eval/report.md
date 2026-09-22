# PG19 position evaluation

All checkpoints: 250,085,376 training tokens. Same 64 validation windows, 16,384 input tokens each. Last-2K evaluation scores output positions [14,336, 16,384), preserving the full causal input. 131,072 scored tokens per model.

| Model | Full-window PPL | Last-8K PPL | Last-2K PPL |
|---|---:|---:|---:|
| gdn-baseline | 30.3579 | 30.1682 | 29.6097 |
| mamba2-baseline | 31.0209 | 30.8499 | 30.3392 |
| mamba2-smat-d2 | 30.9972 | 30.8320 | 30.1853 |
| mamba2-smat-d3 | 31.0146 | 30.8449 | 30.2274 |

mamba2-smat-d2 last-2K PPL change relative to Mamba2 baseline: -0.507%.
mamba2-smat-d3 last-2K PPL change relative to Mamba2 baseline: -0.368%.

This is an exploratory diagnostic on one seed and 64 windows; it does not establish a reliable improvement.

Full-window loss reproduced the recorded milestone validation within 0.000004 absolute NLL for all four checkpoints. Training source hashes and checkpoint configurations were verified. Training runs were not modified.

Initial evaluation app: https://modal.com/apps/archerdwang/main/ap-8wK3H16RmLRWqccUo0QBMi

GDN SMAT d=2/3 have not yet saved their 250M milestones.

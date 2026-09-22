Width32 GDN + SMAT d3: selected configuration, completed32epochs

Final MQAR validation accuracy: 82.77906% versus GDN 69.86219% (+12.91688 percentage points). All15 required gates passed.
The previous configuration (trial22, no scalar decay and incidence rescaling enabled) finished at 81.77875%. The new configuration is 1.00031 percentage points higher. This is a single-seed comparison.

Current configuration: scalar transport decay enabled; incidence rescaling disabled; neighboring write-hash gradients enabled; detached write-hash inputs; content-only one hard bucket write and four distinct learned weighted reads per head. Width32, d3, two layers, two heads, head/state16, seed123, original AdamW LR0.01/WD0.1 and32-epoch cosine schedule, no clipping.

Training paused after epoch14 and resumed from its full model/optimizer/scheduler/RNG checkpoint. Both training phases verified runtime source paths and hashes. Pre-resume audits are archived in pre-resume-epoch14; resume source snapshot and launch record are in resume-epoch14.

| KV pairs | GDN + SMAT | GDN |
|---|---:|---:|
| 4 | 100.00% | 99.98% |
| 8 | 99.84% | 99.71% |
| 16 | 86.19% | 92.91% |
| 32 | 58.20% | 46.49% |
| 64 | 69.67% | 10.22% |

Final resumable checkpoint: w32-d3.pt. Local intermediate checkpoints include04,08,12,14,16,24. All32 immutable epoch checkpoints remain on Modal volume smat-gdn-w32-d3-iterations under /23_decay_no_rescale/checkpoints/.

Final SHA256: ca27647a1e1377eb06d26f0aa9f4994be9e7509bb6171c7ca4c6e144fd7a79e7

Modal app stopped after completion. The historical PAUSED.md records the earlier pause; this FINAL.md supersedes its current-status statement. No PG19 or selective-copying result is implied.

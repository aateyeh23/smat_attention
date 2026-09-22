MQAR width 32, d=3: GDN boundary transport; one write and four reads.

Paused at epoch 23/32 at the user's request. Modal stopped; zero GPU tasks.
Resumable checkpoint downloaded as `w32-d3.pt`; see `paused.json` and `SHA256SUMS`.
Seed 123, LR 0.01, two layers/heads, head/state 16. Training remains incomplete.

| Model at matched epoch | Overall accuracy | 64-pair accuracy | Validation loss |
|---|---:|---:|---:|
| GDN | 68.71% | 9.86% | 1.715 |
| Original SMAT d=3 | 44.05% | 6.02% | 5.972 |
| Transport SMAT d=3 | 54.81% | 3.74% | 3.294 |

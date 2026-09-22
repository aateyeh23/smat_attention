MQAR width 32, d=3: GDN boundary transport; one write and four reads.

Paused at epoch 6/32 at the user's request. Modal stopped, zero tasks.
Checkpoint downloaded with optimizer/scheduler/RNG; see `paused.json`.
Seed 123, LR 0.01, two layers/heads, head/state 16. Training is incomplete.

| Model at matched epoch | Overall accuracy | 64-pair accuracy | Validation loss |
|---|---:|---:|---:|
| GDN | 64.48% | 8.01% | 1.998 |
| Original SMAT d=3 | 45.24% | 4.02% | 3.650 |
| Transport SMAT d=3 | 50.03% | 2.60% | 3.465 |
| Transport + neighbor write gradients | 29.63% | 0.89% | 5.062 |

Neighbor-gradient run started fresh. Original transport paused at epoch 23;
its matched-epoch row is unavailable after epoch 23.

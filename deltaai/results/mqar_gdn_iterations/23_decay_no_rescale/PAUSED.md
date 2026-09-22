MQAR joint ablation: scalar decay restored, profile-output rescaling disabled

Paused at user request after epoch14. Modal app stopped; no Modal containers remained at verification. The 32-epoch run is incomplete.

Everything else matches source-verified trial22: width32, d3, two layers, two heads, head/state16, seed123, one content bucket write, four learned weighted distinct hyperplane reads, neighboring write gradients, detached hash inputs, original single-group AdamW LR0.01 and cosine schedule. Training started from scratch. No CPU model tests were added or run.

| Epoch | Remove both changes | Trial22: retain both | GDN | Gate |
|---|---:|---:|---:|---|
| 4 | 64.02% | 66.66% | 60.66% | pass |
| 6 | 74.14% | 71.45% | 64.48% | pass |
| 8 | 74.86% | 71.50% | 65.50% | pass |
| 10 | 77.07% | 72.96% | 66.12% | pass |
| 12 | 79.90% | 68.74% | 64.92% | pass |
| 14 | 75.92% | 76.01% | 65.65% | pass |

All six required gates through epoch14 passed. At epoch14 the ablation is 0.0953 percentage points below trial22 and 10.2675 points above GDN. Best observed ablation accuracy: 79.89969% at epoch12. These are intermediate single-seed results, not a final32-epoch comparison.

Interpretation: removing the two changes together has not erased the MQAR advantage over GDN, and their necessity is not supported by this partial run. It does not isolate their individual effects or establish PG19/NLP behavior.

Resumable local checkpoints: epochs04,08,12,14. All14 immutable epoch checkpoints remain on Modal volume smat-gdn-w32-d3-iterations under /23_decay_no_rescale/checkpoints/. Epoch14 includes model, optimizer, scheduler and RNG state. No terminal result.json was created because the run was paused.

Latest checkpoint SHA256: 429af23089f6211f16f07b11773559da5492ec307570e886b47bdec7fac4674b

Actual training and GPU-validation source paths and hashes match expected-sources.json and the local launch snapshot.

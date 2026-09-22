**Stopped at user request. All four GPU tasks stopped; partial checkpoints exported to [handoff/README.md](handoff/README.md). The historical log progress below may be ahead of the exported checkpoints.**

PG19 — 2B unique training tokens per model, 16 layers, width 1024, 16K context.

Shared effective batch: 131,072 tokens. Microbatches: 3 + 3 + 2 sequences. Seed 123.
Checkpoints: Modal volume `smat-pg19-2b-checkpoints`, `/seed123/d{1,2,3,4}/`.

| Model | Tokens (M) | Train loss | Validation perplexity | Recent tokens/s | Complete |
|---|---:|---:|---:|---:|---|
| GDN | 899.15 | 2.319 | 21.03 | 106,364 | False |
| GDN + SMAT d=2 | 562.30 | 2.802 | 27.63 | 66,184 | False |
| GDN + SMAT d=3 | 532.15 | 3.060 | 26.29 | 63,835 | False |
| GDN + SMAT d=4 | 433.85 | 3.053 | 28.31 | 51,807 | False |

Latest shared training point: **433.85M tokens**, identical data batches.

| Model | Matched training loss |
|---|---:|
| GDN | 3.024 |
| GDN + SMAT d=2 | 3.160 |
| GDN + SMAT d=3 | 3.056 |
| GDN + SMAT d=4 | 3.053 |

PG19 — 2B unique training tokens per model, 16 layers, width 1024, 16K context.

Shared effective batch: 131,072 tokens. Microbatches: 3 + 3 + 2 sequences. Seed 123.
Checkpoints: Modal volume `pg19-updated-transport-d34`, `/seed123/d{3,4}/`.

| Model | Tokens (M) | Train loss | Validation perplexity | Recent tokens/s | Complete |
|---|---:|---:|---:|---:|---|
| GDN + SMAT d=3 | 36.70 | 4.877 | — | 33,010 | False |
| GDN + SMAT d=4 | 22.28 | 5.459 | — | 20,532 | False |

Latest shared training point: **22.28M tokens**, identical data batches.

| Model | Matched training loss |
|---|---:|
| GDN + SMAT d=3 | 5.451 |
| GDN + SMAT d=4 | 5.459 |

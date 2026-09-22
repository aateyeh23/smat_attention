# PG19 six-run comparison

Updated: 2026-09-18T00:13:34.363882+00:00

16 layers, width 768, FFN 2048, 16K context. One H100 per arm. Seed 123.
Plain baselines retain full-sequence recurrence; SMAT arms reset at the midpoint.

| Model | Phase | Tokens (M) | Train loss | Latest val PPL (tokens M) | Tokens/s |
|---|---|---:|---:|---:|---:|
| gdn-baseline | complete | 500.01 | 3.184 | 24.43 (500.01) | 110,981 |
| gdn-smat-d2 | complete | 500.01 | 3.182 | 24.36 (500.01) | 53,712 |
| gdn-smat-d3 | complete | 500.01 | 3.181 | 24.46 (500.01) | 42,633 |
| mamba2-baseline | complete | 500.01 | 3.223 | 25.29 (500.01) | 60,217 |
| mamba2-smat-d2 | complete | 500.01 | 3.221 | 25.32 (500.01) | 59,027 |
| mamba2-smat-d3 | complete | 500.01 | 3.228 | 25.39 (500.01) | 39,041 |

## Validation at 500.01M matched training tokens

| Model | Perplexity |
|---|---:|
| gdn-baseline | 24.43 |
| gdn-smat-d2 | 24.36 |
| gdn-smat-d3 | 24.46 |
| mamba2-baseline | 25.29 |
| mamba2-smat-d2 | 25.32 |
| mamba2-smat-d3 | 25.39 |

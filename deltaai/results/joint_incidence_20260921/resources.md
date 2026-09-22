# Resource accounting

Profiles and types below are per head per layer. Matrix bytes sum both layers and both heads for one example. Trainable parameters cover the full model, including the length-specific routers. Timings use length 3076 and batch 256 on NVIDIA GH200 120GB.

| Variant | Trainable parameters | Profiles | Read types | Profile + summary bytes | Read-table bytes | Measured train step (ms) |
|---|---:|---:|---:|---:|---:|---:|
| geometry | 193064 | 169 | 182 | 1437696 | 745472 | 134.0 |
| random-17 | 193064 | 169 | 182 | 1437696 | 745472 | 133.3 |
| random-29 | 193064 | 169 | 182 | 1437696 | 745472 | 133.2 |
| buckets_profiles | 181364 | 169 | 169 | 692224 | 692224 | 131.3 |
| buckets_bytes | 312924 | 351 | 351 | 1437696 | 1437696 | 161.0 |

The byte-matched control has 351 independently addressable states versus 169 profiles plus 182 derived summaries. It receives more independent payload capacity at the same combined matrix-table budget. Its reader is correspondingly larger. The primary geometry/random comparison has identical parameter shapes and every table size.

These table budgets are not total training allocations or measurements of an autoregressive cache. Full-batch peak allocations and all five lengths are retained in validated.json.

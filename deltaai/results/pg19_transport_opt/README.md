# PG19 GDN + SMAT performance optimization — September 17, 2026

Implemented and validated `tiled_fp32_v1` on H100. The user subsequently
authorized resuming both production GDN + SMAT runs with these kernels.
Their original paused checkpoints and source manifests were retained;
checksum-verified copies are continuing in `six-500m-20260917-opt-v1`.
The GDN baseline and all three Mamba2 runs continue in the original app.

## Full-model results

Same saved weights, real PG19 rows, 16 layers, width 768, 16K context,
microbatch 2, six 128-dimensional heads, rank-64 routing. Paired numbers are
median forward/backward times over three repetitions on the same H100.
Optimizer pilots resume isolated copies of each checkpoint for five steps,
with the original global batch of eight sequences, optimizer state, data
cursor, RNG state and 500M-token schedule. Steady pilot speed excludes step one.

| GDN + SMAT | Original forward/backward tok/s | Optimized forward/backward tok/s | Paired speedup | Optimized optimizer-step tok/s |
|---|---:|---:|---:|---:|
| d=2 | 24,454 | 55,801 | 2.28x | 55,617 |
| d=3 | 20,302 | 42,768 | 2.11x | 42,349 |

Pilot peak allocated memory was 59.28 GB for d=2 and 74.08 GB for d=3.
These are bounded pilots, not a claim about long-run validation quality.

## Changes

- `smat_read_tiled.py`: 64x64 state tiles for the FP32 reader, avoiding the
  slow 128x128 launch shape. Full-size reader forward/backward improved from
  26.6/31.0 ms to 2.49/3.11 ms for d=2/3 synthetic routing layouts.
- `smat_gdn_tiled.py`: an opt-in FLA backend keeps the existing GDN backward
  kernel and equations but selects 64x64 key/value tiles for FP32 transport
  on H100. The layer profile's combined `chunk_bwd_kernel_dqkwg` time fell
  from 22.46 ms to 1.59 ms.
- `smat_write_hash_tiled.py`: group neighbor hash-gradient contractions by
  memory cell and reuse its tiles. Full-size d=2/3 contractions improved
  from 2.45/4.89 ms to 0.58/0.99 ms.
- `pg19_optimized_kernels.py`: shape-guarded opt-in dispatch, with original
  kernels retained for other shapes. No parameters or checkpoint keys change.

Transport still uses FP32 and 16-token chunks. Benchmarks of 32/64-token
chunks did not provide a useful layer speedup, so they were not selected.
The model equations, directional transport, routing, write gradients, state
dimensions, loss, training examples, optimizer and learning-rate schedule
remain unchanged. Floating-point summation order changes; results are not
bitwise identical.

## Verification

Reader and write-gradient kernels were compared with the original kernels
at both small correctness shapes and the full production shapes. Reader
outputs/gradients and write gradients had approximately 2–3e-7 relative L2
differences. The reader also passed an explicit dense output/gradient reference.
Long 16K transport tests covered weak, moderate and strong decay; unchanged
forward outputs and relative gradient errors below 5e-7 at the selected
16-token chunk size were observed.

On trained full-model checkpoints:

| d | Absolute language-loss difference | Relative L2 difference over all trainable gradients |
|---|---:|---:|
| 2 | 0.0000553 | 0.456% |
| 3 | 0.00000143 | 0.137% |

All full-model gradients were finite; both five-step optimizer continuations
passed and saved resumable checkpoints in the separate benchmark volume.
Small per-parameter gradients can have larger relative differences; detailed
values are retained in `summary.json` and `final-validation.log`.
Python compilation and `git diff --check` passed.

Final full validation app: https://modal.com/apps/archerdwang/main/ap-LQRWIAH5nCLVMq77iERKAb
Results volume: `pg19-transport-optimization:/validate-final-chunk16/`.
The experimental pilot checkpoints there must not be confused with the paused
production checkpoints.

## Production state and opt-in entrypoint

Original volume: `pg19-w768-rank64:/seed123/six-500m-20260917/`.

| Arm | Saved step | Saved tokens | Checkpoint size |
|---|---:|---:|---:|
| gdn-smat-d2 | 301 | 39,452,672 | 1,996,593,373 bytes |
| gdn-smat-d3 | 264 | 34,603,008 | 2,058,956,477 bytes |

`paused-checkpoints.json` records remote verification of both paused states.
Optimized production continuations were launched September 17 at 16:03 CDT
in app `ap-IeUwYiy5yeh7EdOo5GJEWm`, from these exact saved steps. The original
directories remain as provenance; their `continuation.json` records redirect
the six-arm collector to the new campaign.

`modal_pg19_six.py::run` now accepts `--optimized` for GDN d=2/3 only. It
freezes and audits the additional kernels through `lm/run_pg19_optimized.py`
and records `kernel_optimization=tiled_fp32_v1`. Existing source-manifest
mismatch protection remains intact. Resuming a paused original run with the
optimization requires an explicit, documented checkpoint migration to an
optimized campaign directory; `resume_gdn_optimized` performs that migration
and was used for this continuation. Do not bypass or remove the protection.

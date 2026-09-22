# Learned-routing models

GDN and Mamba-2 with SMat memory: each distant token writes to one hashed
profile, each recent query reads four incidence summaries chosen by a learned
scorer (`docs/practical_model_revision.md` defines the model).  These are the
models behind the PG-19 results and the current recall configurations.

The modules import each other by flat name, as they did when they ran, and are
byte-identical to the files the PG-19 six-run campaign recorded:
`results/pg19_six_500m/<arm>/manifest.json` lists their SHA-256.  Put this
directory, `tasks/pg19` and `third_party` on `PYTHONPATH` (see the top-level
README) instead of importing them as a package.

| module | role |
|---|---|
| `zoo_smat_gdn.py`, `zoo_gdn_transport.py`, `smat_gdn_transport.py` | GDN + SMat mixer, boundary-transported profile memory |
| `zoo_smat_mixer.py` | Mamba-2 + SMat mixer |
| `content_addr.py`, `smat_address_reads.py` | write hash and four-read selection |
| `smat_read_triton.py`, `smat_write_hash_triton.py`, `smat_write_hash_grad.py`, `smat_delta_pool.py`, `smat_pool_ops.py` | fused reads, hashed writes and their gradients |
| `smat_read_tiled.py`, `smat_write_hash_tiled.py`, `smat_gdn_tiled.py`, `pg19_optimized_kernels.py` | tiled FP32 kernels the GDN + SMat PG-19 arms trained with (`pg19_optimized_kernels.enable()`) |
| `gdn_smat_scale.py`, `gdn_smat_transport_scale.py`, `mamba_smat_scale.py`, `pg19_baselines.py` | the PG-19 language models and their plain baselines |
| `pg19_recurrent.py` | cached per-token decoding, checked against full-window logits by `tasks/pg19/test_pg19_recurrent_gpu.py` |
| `zoo_log_linear.py` | dense Log-Linear reference used to check the upstream kernels |
| `smat_mask.py`, `smat_attn.py`, `smat_triton.py` | flat copies of `src/smat/{mask,attention,kernels/triton_fwd}.py` (the first and last byte-identical, the second differing only in import names): `zoo_smat_mixer.py` imports them by these names |

The recall tables (2-3) ran earlier versions of several of these files; those
versions are frozen with their results in `results/paper_seeds_20260920/source/`.

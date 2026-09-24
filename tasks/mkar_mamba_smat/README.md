# Learned Mamba-2 + SMat on multi-key subset recall

The campaign scripts behind `results/mkar_mamba_smat_20260923/`.  The model
shell, data, loss and evaluator are `tasks/mkar.py`'s; only the sequence mixer
is replaced (`learned_mixers.py`, which builds `src/smat_lm/zoo_mamba_four_reads.py`).

| script | role |
|---|---|
| `common.py` | the fixed protocol, paths, and the source check |
| `campaign_model.py`, `learned_mixers.py` | the model |
| `validate.py` | pre-launch GPU checks: data semantics, common initialization, gradient isolation of the write address, checkpoint resume; writes `validated.json` |
| `train.py --index i` | trains task `i` of `tasks.json` (one seed, k = 1, 2, 3 in turn); resumable, exits 75 when out of time |
| `collect.py` | `results.csv`, `summary.json`, `summary.md` from the finished fits |
| `prepare.py` | how the confirmation campaign was frozen from the pilot (needs the pilot's frozen source tree, not kept here) |

The runs used PyTorch 2.11 (CUDA 13), Triton 3.7.1, `mamba-ssm`,
`causal-conv1d` and `flash-linear-attention` as in the top-level README, on one
GPU in FP32.  They ran from a frozen copy of the modules they import; every
one of those is byte-identical here, except for a scrubbed comment in
`src/smat_lm/zoo_smat_mixer.py`.  Without the frozen copy, `common.py` puts the
repository modules on the path and checks each against the recorded manifest.
`SCRUBBED.json` lists the three scripts edited for this repository.

To rerun the four confirmation seeds:

```bash
experiments/submit.sh experiments/jobs/run_mkar_mamba_smat.sbatch validate
for i in 0 1 2 3; do experiments/submit.sh experiments/jobs/run_mkar_mamba_smat.sbatch train $i; done
```

Runs go to `$SMAT_WORK/mkar_mamba_smat/confirm` (`MKAR_ROOT` overrides), which
starts from the recorded `protocol.json`, `tasks.json` and `source_sha256.json`.
Set `MKAR_CAMPAIGN=pilot_s123` for the selection seed (task 0 only).

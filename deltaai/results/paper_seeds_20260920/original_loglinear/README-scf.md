# Running the six arms on SCF instead of Modal

`modal_log_linear.py` is the campaign of record, and it targets Modal. This
account has no Modal credentials (`~/.modal.toml` absent, `modal` not
installed), and the campaign was left `paused_by_user` with the GDN arms
stopped before their first epoch checkpoint. SCF has Slurm GPUs, so the same
code runs here through `sbatch`.

**Nothing that decides a number changed.** `zoo_log_linear.py` (the operator),
`zoo_reference_configs.py` (the MQAR mixture), `zoo_mqar_resume.py` and
`zoo_mqar_interleave.py` (the training recipe) are byte-identical to the Modal
run. The six arms, their learning rates (0.01, except GDN width 64 at 0.003),
seed 123, 32 epochs, head and state dimension 16, GDN heads 1/2/2 and Mamba-2
heads 2/4/8 come from the same `zoo_log_linear_configs.py`.

Three things had to move:

1. **`zoo_log_linear_configs.py`** pinned the operator source to
   `/opt/log-linear/zoo_log_linear.py`, the path inside the Modal image. The
   assertion now reads that path from `LL_SOURCE_PATH` and keeps the Modal path
   as its default, so the SHA-256 audit still gates training on the exact file
   the GPU validation pass checked.
2. **`SmatMamba2Block`** lives in `zoo_smat_mixer.py`, which imports
   `mamba_ssm` and `causal_conv1d` at module scope; neither is installed on SCF,
   as both need an nvcc build the compute nodes do not expose. Two import-only
   shims on `PYTHONPATH` (see below) make that module import as it does on
   Modal, so the arms take the original path. `zoo_log_linear_block.py` holds a
   character-for-character copy of the block as a fallback, used only if
   `zoo_smat_mixer` ever fails to import; the run log records which one was
   taken (`Mamba2Block -> SmatMamba2Block zoo_smat_mixer` is the original).
3. **`test_log_linear_gpu.py`** read the upstream reference from
   `/opt/log-linear/upstream_base.py`; it now reads `LL_UPSTREAM_BASE`, again
   with the Modal path as the default. `upstream/hattention/base.py` here is
   `HanGuo97/log-linear-attention@7f8644159c1406fae1ad863829a5b3a4fbf63022`,
   the commit the manifest names.

## Two import-only shims

Both live in `/scratch/users/abdullah_ateyeh/ll-shims` and neither supplies an
implementation of anything:

- **`mamba_ssm`** sets `__path__` to Zoology's own vendored copy of those Triton
  kernels (`zoology/mixers/mamba_ssm/`). The vendored files kept their original
  absolute imports, so the name has to resolve for `zoology/mixers/mamba2.py` to
  import at all. This points it at Zoology's own source, not a substitute.
- **`causal_conv1d`** answers the import in `zoology/mixers/mamba2.py` line 24.
  That file never calls the symbol -- line 24 is its only mention outside the
  assert that follows -- so every entry point in the shim raises instead of
  computing. A Log-Linear arm that ever reached one would die rather than take a
  different convolution than the Modal campaign took.

Neither kernel path is on a Log-Linear arm's route: `LogLinearMixer` reads
Mamba-2's projections and convolution directly and runs its own dense base-2
recursion, and `Mamba2` is constructed with `use_mem_eff_path=False`.

## Environment

Zoology is not on PyPI and was not cloned on this machine; it is
`HazyResearch/zoology@1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb` at
`/scratch/users/abdullah_ateyeh/zoology`, on `PYTHONPATH` rather than
installed, so its `setup.py` does not drag in `causal_conv1d`. The Python
environment is `/scratch/users/abdullah_ateyeh/venvs/loglinear`, a venv over
the existing `base311` conda environment (torch 2.11.0+cu128, triton 3.6.0,
flash-linear-attention 0.5.2, einops) with `pydantic`, `pandas`, `wandb` and
`torchvision` 0.26.0 added -- the last only because `zoology/model.py` imports
`StochasticDepth`. Weights & Biases is disabled; Zoology's logger is inert
without an entity.

## Operation

    experiments-style submit, from deltaai/:
    sbatch --job-name=ll-validate run_log_linear_validate.sbatch
    sbatch --job-name=ll-gdn-w16  run_log_linear_arm.sbatch gdn 16     # and the other five

The validation gate writes `validation/passed.json`; each arm asserts that the
gate covers its operator SHA before training, exactly as the Modal version did.
Arms checkpoint every epoch into `{family}-w{width}/` with per-epoch snapshots
under `checkpoints/`, so a job that hits its walltime resumes where it stopped
(exit code 3 means "time limit, resubmit"). `status_log_linear.py` prints the
six arms from disk, in place of `collect_log_linear.py`, which pulls the same
fields out of the Modal volume.

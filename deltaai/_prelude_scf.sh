# Berkeley SCF version of _prelude.sh.  The DeltaAI prelude loads an NCSA module
# (python/miniforge3_pytorch/2.11.0) that does not exist here; SCF has no such
# module, so we put the existing conda env on PATH instead.  Everything else --
# the cd, the PYTHONPATH with $PWD first for sitecustomize.py, and the
# host/GPU/toolchain banner -- is unchanged.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[1]}")}"
export PATH="/scratch/users/abdullah_ateyeh/conda/envs/215a/bin:$PATH"
export SMAT="$(cd ../smat && pwd)"
export PYTHONPATH="$PWD:$SMAT${PYTHONPATH:+:$PYTHONPATH}"
echo "host: $(hostname)   started: $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import torch; print('torch', torch.__version__,
      'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
try:
    import triton; print('triton', triton.__version__)
except ImportError:
    print('triton: not installed')"

# Fail fast: nvidia-smi can see a GPU on a node where this torch build cannot use
# it (luthien, job 3529258: A100 visible, torch.cuda.is_available() False, silent
# CPU fallback at 4 s/step).  Never let a sweep run on CPU by accident.
python - <<'PYEOF' || { echo "NO USABLE GPU on $(hostname) -- aborting"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    sys.exit(1)
cap = torch.cuda.get_device_capability(0)
archs = [int(a[3:]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
if cap[0] * 10 + cap[1] not in archs:
    print(f"device cc {cap} not in {archs}"); sys.exit(1)
PYEOF

echo

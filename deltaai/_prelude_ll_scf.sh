# Berkeley SCF environment for the six MQAR Log-Linear arms.
#
# The campaign was written for Modal (modal_log_linear.py), which is not
# configured on this account: no token, no ~/.modal.toml.  SCF has Slurm GPUs
# instead, so the same code runs here through sbatch.  Everything that decides a
# number -- zoo_log_linear.py, zoo_log_linear_configs.py, zoo_mqar_resume.py,
# zoo_mqar_interleave.py, zoo_reference_configs.py -- is untouched; only the
# execution environment is local.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[1]}")}"
export LL_VENV=/scratch/users/abdullah_ateyeh/venvs/loglinear
export LL_ZOOLOGY=/scratch/users/abdullah_ateyeh/zoology
export LL_SHIMS=/scratch/users/abdullah_ateyeh/ll-shims
export PATH="$LL_VENV/bin:$PATH"
export SMAT="$(cd ../smat && pwd)"
export PYTHONPATH="$PWD:$SMAT:$LL_ZOOLOGY:$LL_SHIMS${PYTHONPATH:+:$PYTHONPATH}"
export LL_SOURCE_PATH="$PWD/zoo_log_linear.py"
export LL_UPSTREAM_BASE="$PWD/results/mqar_log_linear/upstream/hattention/base.py"
export WANDB_MODE=disabled FLA_TILELANG=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/results/mqar_log_linear/triton-cache/${SLURM_JOB_ID:-local}"
mkdir -p "$TRITON_CACHE_DIR"
echo "host: $(hostname)   started: $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
try:
    import triton; print('triton', triton.__version__)
except ImportError:
    print('triton: not installed')"

# Same fail-fast as _prelude_scf.sh: never let an arm run on CPU by accident.
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

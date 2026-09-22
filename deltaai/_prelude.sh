# DeltaAI (NCSA) version of ../smat/_prelude.sh.  Sourced by every job script
# in this folder.  Puts the job in this directory, loads the site PyTorch env
# (torch 2.11 + triton 3.5.1 + matplotlib), and records host/GPU/toolchain.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[1]}")}"
module load python/miniforge3_pytorch/2.11.0
export SMAT="$(cd ../smat && pwd)"       # the untouched repo code
export PYTHONPATH="$PWD:$SMAT${PYTHONPATH:+:$PYTHONPATH}"   # $PWD first: sitecustomize.py shim
echo "host: $(hostname)   started: $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import torch; print('torch', torch.__version__,
      'cuda', torch.cuda.is_available())
try:
    import triton; print('triton', triton.__version__)
except ImportError:
    print('triton: not installed')"
echo

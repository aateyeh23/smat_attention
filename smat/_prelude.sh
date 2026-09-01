# Sourced by every job script.  Puts the job in the submit directory, then says
# which host, GPU and toolchain produced the numbers below it -- a sweep whose
# log does not record the driver and torch version cannot be compared against
# another one later.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[1]}")}"
echo "host: $(hostname)   started: $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import torch; print('torch', torch.__version__,
      'cuda', torch.cuda.is_available())
try:
    import triton; print('triton', triton.__version__)
except ImportError:
    print('triton: not installed')"
echo

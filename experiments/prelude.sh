# Sourced by every job script.  Records what produced the numbers -- library
# versions and the device's compute capability -- and nothing that identifies
# the machine or the site: a results file should be publishable as it stands.
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
[ -f experiments/site.conf ] && . experiments/site.conf
[ -n "${SMAT_ENV:-}" ] && eval "$SMAT_ENV"
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
if torch.cuda.is_available():
    c = torch.cuda.get_device_capability()
    g = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device sm_{c[0]}{c[1]}  {g:.0f} GiB")     # capability, not a model name
try:
    import triton; print(f"triton {triton.__version__}")
except ImportError:
    print("triton: not installed (torch fallbacks will be used)")
PY
echo

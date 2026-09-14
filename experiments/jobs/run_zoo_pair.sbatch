#!/bin/bash
#SBATCH --job-name=zoo-pair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Two Zoology sweeps sharing one GPU inside a single job:
#   A: hybrid d=2 (Mamba-2 + G, no reset), lrs 3.16e-3 / 1e-2 / 3.16e-2 -> logs/zoo-hybrid-d2-b.out
#   B: SMAT with reset, d=3 then d=1, 4 lrs                             -> logs/zoo-reset-d3d1.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_DM=16 ZOO_EPOCHS=32
echo "==== diagnostics ===="; which python
python - <<'PY' 2>&1 | grep -vi warn
try:
    import causal_conv1d; print("causal_conv1d OK", causal_conv1d.__file__)
except Exception as e: print("causal_conv1d FAILED:", type(e).__name__, str(e)[:300])
import mamba_ssm.modules.mamba2 as m2; print("mamba2 has causal_conv1d_fn:", m2.causal_conv1d_fn is not None)
PY
FILT='grep --line-buffered -vi warn | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
( ZOO_DS=2 ZOO_RESET=0 ZOO_LRS=3.16e-3,1e-2,3.16e-2 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/zoo-hybrid-d2-b.out 2>&1 ) &
( ZOO_DS=3,1 ZOO_RESET=1 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/zoo-reset-d3d1.out 2>&1 ) &
wait
echo "done: $(date)"

#!/bin/bash
#SBATCH --job-name=zoo-rest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Zoology pipeline, width 16: Mamba-2 reference (4 lrs), then SMAT d=3 and d=1.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_DM=${ZOO_DM:-16} ZOO_EPOCHS=32
F='grep --line-buffered -vi warn | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
echo "==== mamba2 ===="; ZOO_MODELS=mamba2 python -m zoology.launch experiments/zoology/reference.py 2>&1 | eval "$F"
echo "==== smat d=3,1 ===="; ZOO_DS=3,1 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$F"
echo "done: $(date)"

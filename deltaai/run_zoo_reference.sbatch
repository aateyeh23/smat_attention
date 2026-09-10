#!/bin/bash
#SBATCH --job-name=zoo-ref
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/zoo-ref-%j.out

# Reference numbers from Zoology's own pipeline (mixture training, per-cell
# test, lr sweep, early stop): attention and Mamba-2 at d_model ZOO_DM.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_DM=${ZOO_DM:-16} ZOO_EPOCHS=32
python -m zoology.launch zoo_reference_configs.py 2>&1 | grep --line-buffered -vi warn | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E 's/.*(valid\/loss=.*)\]$/\1/' | grep --line-buffered -vE "^\s*$" | awk '!seen[$0]++ {print; fflush()}'
echo "done: $(date)"

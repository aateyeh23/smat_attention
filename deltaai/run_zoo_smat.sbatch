#!/bin/bash
#SBATCH --job-name=zoo-smat
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/zoo-smat-%j.out

# SMAT arms (Mamba-2 + G, d=2,3, mask bank; d=1 consistency check) through
# Zoology's own MQAR pipeline: mixture training, per-cell test, lr sweep,
# early stop.  Width ZOO_DM.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_DM=${ZOO_DM:-16} ZOO_EPOCHS=32 ZOO_DS=${ZOO_DS:-2,3,1}
python -m zoology.launch zoo_smat_configs.py 2>&1 | grep --line-buffered -vi warn | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E 's/.*(valid\/loss=.*)\]$/\1/' | grep --line-buffered -vE "^\s*$" | awk '!seen[$0]++ {print; fflush()}'
echo "done: $(date)"

#!/bin/bash
#SBATCH --job-name=prof
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-}
cd /u/an author/smat_attention/the GPU cluster; python -u analysis/profile_layer.py 2>&1 | grep -v -i "warn\|hi mamba\|custom_\|torchao"; echo PROF DONE

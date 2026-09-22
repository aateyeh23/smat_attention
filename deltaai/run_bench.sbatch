#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:25:00
#SBATCH --output=logs/bench-%j.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/archerdw/smat_attention/deltaai; python -u lm/bench_mixer2.py 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao"; echo BENCH DONE

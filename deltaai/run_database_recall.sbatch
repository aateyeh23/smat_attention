#!/bin/bash
#SBATCH --job-name=database-recall
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/logs/database-recall-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention/deltaai
export PYTHONPATH="/u/archerdw/triton371:$PWD:$PWD/../smat:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/results/synthetic_memory/triton_cache"
python -u database_recall.py "$@"

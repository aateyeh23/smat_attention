#!/bin/bash
#SBATCH --job-name=mkar-mamba-normalized-stop-pilot-check
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/results/learned_mkar_original_mamba_transport_normalized_stop_20260923/logs/check-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention
MKAR_ROOT="$PWD/deltaai/results/learned_mkar_original_mamba_transport_normalized_stop_20260923"
export PYTHONSAFEPATH=1 FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$MKAR_ROOT/source/experiment:$MKAR_ROOT/source/original/src:$MKAR_ROOT/source/original/tasks:/u/archerdw/triton371:$MKAR_ROOT/source/deltaai:$MKAR_ROOT/source/smat:$MKAR_ROOT/source:/u/archerdw/.local/lib/python3.12/site-packages"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/mkar-k123-check-${USER}-${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
python -P -u "$MKAR_ROOT/source/experiment/validate.py"

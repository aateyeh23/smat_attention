#!/bin/bash
#SBATCH --job-name=mom-validate
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/results/joint_mom_20260921/logs/validate-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention
export PYTHONSAFEPATH=1 FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/deltaai/mom_ablation:/u/archerdw/triton371:$PWD/deltaai/results/joint_incidence_20260921/source/deltaai:$PWD/deltaai/results/joint_incidence_20260921/source/smat:$PWD/deltaai/results/joint_incidence_20260921/source:/u/archerdw/.local/lib/python3.12/site-packages"
export TRITON_CACHE_DIR="$PWD/deltaai/results/synthetic_memory/triton_cache"
python -P -u deltaai/mom_ablation/validate.py

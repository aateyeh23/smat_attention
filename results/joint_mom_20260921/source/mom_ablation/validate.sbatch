#!/bin/bash
#SBATCH --job-name=mom-validate
#SBATCH --account=<account>
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=results/joint_mom_20260921/logs/validate-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd .
export PYTHONSAFEPATH=1 FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/mom_ablation:<triton-3.7.1>:$PWD/results/joint_incidence_20260921/source/the GPU cluster:$PWD/results/joint_incidence_20260921/source/smat:$PWD/results/joint_incidence_20260921/source:<site-packages>"
export TRITON_CACHE_DIR="$PWD/results/synthetic_memory/triton_cache"
python -P -u mom_ablation/validate.py

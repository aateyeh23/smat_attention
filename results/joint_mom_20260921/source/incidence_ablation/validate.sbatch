#!/bin/bash
#SBATCH --job-name=incidence-validate
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=results/joint_incidence_20260921/logs/validate-%j.out
set -eo pipefail
module load <pytorch-2.11-module> >/dev/null 2>&1
cd .
export PYTHONSAFEPATH=1
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/incidence_ablation:<triton-3.7.1>:$PWD/results/joint_incidence_20260921/source/the GPU cluster:$PWD/results/joint_incidence_20260921/source/smat:$PWD/results/joint_incidence_20260921/source:<site-packages>"
export TRITON_CACHE_DIR="$PWD/results/synthetic_memory/triton_cache"
python -P -u incidence_ablation/validate.py

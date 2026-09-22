#!/bin/bash
#SBATCH --job-name=paper-ll-tree
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/paper_seeds/optimized_probe/tree-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention
export PYTHONPATH="$PWD/deltaai/paper_seeds/optimized_probe:$PWD/deltaai/results/paper_seeds_20260920/python_deps:$PWD/deltaai/paper_seeds:/u/archerdw/triton371:$PWD/deltaai/results/paper_seeds_20260920/source/joint/deltaai:$PWD/deltaai/results/paper_seeds_20260920/source/joint/smat:$PWD/deltaai/results/paper_seeds_20260920/source/joint:/u/archerdw/.local/lib/python3.12/site-packages"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/$USER-loglinear-speed-probe" PYTHONSAFEPATH=1


for family in mamba2 gdn; do
  python -P -u deltaai/paper_seeds/optimized_probe/probe.py --family "$family" --backend tree && python -P -u deltaai/paper_seeds/optimized_probe/benchmark.py --family "$family" --backend tree
done

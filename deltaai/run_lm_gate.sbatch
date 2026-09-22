#!/bin/bash
#SBATCH --job-name=smat-lm-gate
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=01:00:00
#SBATCH --output=logs/lm-gate-%j.out

# Gate 1: input-dependent key gate on the landmark write, d=2, T=4096, QK-norm,
# fused kernels.  Compare to smat_d2_qkn 2.44 and the no-long-range control 2.20.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
echo "==== arm: --arm smat --d 2 --gate   $(date) ===="
OMP_NUM_THREADS=4 python -u train_lm.py --arm smat --d 2 --gate --qk-norm --triton-bwd \
    --T 4096 --layers 4 --d-model 256 --heads 4 --r 64 \
    --steps 20000 --nb 8 --lr 3e-4 --warmup 500 --eval-every 2000 --val-windows 128 \
    --out results/lm_T4096_20k_qkn_gate.csv
echo "done: $(date)"

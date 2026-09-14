#!/bin/bash
#SBATCH --job-name=smat-lm-8k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# T=8192, QK-norm, 20k steps; SMAT arms train through the fused Triton backward
# kernels (smat_triton_bwd.py).  Same model and optimiser as the T=4096 runs.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"

for ARM in "--arm softmax" "--arm smat --d 1" "--arm smat --d 2" "--arm smat --d 3"; do
    echo "==== arm: $ARM   $(date) ===="
    OMP_NUM_THREADS=4 python -u src/smat/model.py $ARM --qk-norm --triton-bwd \
        --T 8192 --layers 4 --d-model 256 --heads 4 --r 64 \
        --steps 20000 --nb 8 --lr 3e-4 --warmup 500 --eval-every 2000 --val-windows 128 \
        --out results/lm_T8192_20k_qkn_tri.csv
done
echo "done: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-lm-nolr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Control: the d=2 landmark/recent split with the long-range block removed, at
# T=4096 and T=8192, QK-norm, fused kernels.  Compare to smat_d2_qkn (2.44 at
# 4096) and smat_d1_qkn (2.98): if this matches d=2, the LM gain is sparsity.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
for T in 4096 8192; do
    echo "==== arm: --arm smat --d 2 --no-lr T=$T   $(date) ===="
    OMP_NUM_THREADS=4 python -u src/smat/model.py --arm smat --d 2 --no-lr --qk-norm --triton-bwd \
        --T $T --layers 4 --d-model 256 --heads 4 --r 64 \
        --steps 20000 --nb 8 --lr 3e-4 --warmup 500 --eval-every 4000 --val-windows 128 \
        --out results/lm_T${T}_20k_qkn_nolr.csv
done
echo "done: $(date)"

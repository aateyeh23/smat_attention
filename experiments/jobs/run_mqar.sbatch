#!/bin/bash
#SBATCH --job-name=smat-mqar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# MQAR (Zoology layout) at T=512: 16 kv pairs in [0,32), 16 queries in the
# recent half [256,512), 128 keys, random noise elsewhere.  Softmax solves this
# in ~1000 steps; at T>=1024 it did not converge in this budget.  QK-norm,
# causal short conv (width 3) in every block, fused kernels.  d_max(512)=3.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
for ARM in "--arm softmax" "--arm smat --d 1" "--arm smat --d 2" "--arm smat --d 3"; do
    echo "==== arm: $ARM   $(date) ===="
    OMP_NUM_THREADS=4 python -u tasks/mqar.py $ARM --qk-norm --short-conv 3 --triton-bwd \
        --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 \
        --layers 4 --d-model 256 --heads 4 --r 64 \
        --steps 2500 --nb 32 --lr 1e-3 --eval-every 500 --eval-n 256 \
        --out results/mqar_T512.csv
done
echo "done: $(date)"

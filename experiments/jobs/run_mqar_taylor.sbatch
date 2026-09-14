#!/bin/bash
#SBATCH --job-name=smat-mqar-taylor
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# MQAR T=512, 16 pairs: SMAT d=1 with the Taylor (Based) feature map, m=16 ->
# 273 features.  Torch path (r > 128 is outside the fused backward kernels).
# Reference: elu+1 d=1 scored 1.3%, softmax 100% (results/mqar_T512.csv).
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
echo "==== arm: --arm smat --d 1 --phi taylor   $(date) ===="
OMP_NUM_THREADS=4 python -u tasks/mqar.py --arm smat --d 1 --phi taylor --taylor-dim 16 --qk-norm --short-conv 3 \
    --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 \
    --layers 4 --d-model 256 --heads 4 \
    --steps 2500 --nb 32 --lr 1e-3 --eval-every 500 --eval-n 256 \
    --out results/mqar_T512.csv
echo "done: $(date)"

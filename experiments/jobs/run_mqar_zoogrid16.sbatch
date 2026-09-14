#!/bin/bash
#SBATCH --job-name=smat-zoo16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Log-Linear Table-2 style MQAR at width 16: T=256, 4..64 kv pairs (Zoology
# generator, vocab 8192), 2 layers, batch 256, 32 epochs of 20k examples
# (2500 steps), lr in logspace(-3,-1.5,4), early stop at 99%, seed 0.
# Arms: real Mamba-2 mixer; Mamba-2 mixer + SMAT landmark block G at d=2,3
# (segment reset via seq_idx, per-head mask-bank offsets, no global channel).
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
OUT=results/mqar_zoogrid_dm16.csv
for P in 4 8 16 32 64; do
  for LR in 0.001 0.00316 0.01 0.0316; do
    for ARM in "--arm mamba2" "--arm mamba2smat --d 2 --mask-bank" "--arm mamba2smat --d 3 --mask-bank"; do
      echo "==== T=256 pairs=$P lr=$LR   arm: $ARM   $(date) ===="
      OMP_NUM_THREADS=4 python -u tasks/mqar.py $ARM --zoology --T 256 --n-pairs $P --chunk 64 --layers 2 --d-model 16 --heads 1 --r 64 --qk-norm --short-conv 0 \
          --nb 256 --lr $LR --warmup 100 --steps 2500 --eval-every 250 --eval-n 512 --early-stop 0.99 --out $OUT --tag _lr$LR
    done
  done
done
echo "done: $(date)"

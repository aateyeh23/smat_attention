#!/bin/bash
#SBATCH --job-name=smat-mqar-tab
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/mqar-table-%j.out

# Standard MQAR (Zoology generator, T=256, 4..64 kv pairs, vocab 8192), 2-layer
# models at d_model 16/32/64 (head dim 16 -> 1/2/4 heads, r=64), early stop at
# 99%, seeds outermost.  Arms: gated d=1 (Mamba-2-like reference); gated d=2 and
# d=3 with the global channel over the context (n_P = 2*pairs, capped by the
# builder at 64 pairs) and a mask bank (distinct row-type offset per layer/head).
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
COMMON="--zoology --T 256 --chunk 64 --layers 2 --r 64 --decay mamba2 --dt-min 0.05 --dt-max 0.5 --A-min 1 --A-max 2 --gate-wd 0 --qk-norm --short-conv 3 --steps 8000 --nb 64 --lr 1e-3 --eval-every 500 --eval-n 512 --early-stop 0.99 --out results/mqar_table_T256.csv"
for SEED in 0 1 2; do
  for DM in 16 32 64; do
    H=$((DM / 16))
    for P in 4 8 16 32 64; do
      for ARM in "--arm smat --d 1" "--arm smat --d 2 --n-P -1 --mask-bank" "--arm smat --d 3 --n-P -1 --mask-bank"; do
        echo "==== seed=$SEED dm=$DM pairs=$P   arm: $ARM   $(date) ===="
        OMP_NUM_THREADS=4 python -u mqar_lm.py $ARM $COMMON --d-model $DM --heads $H --n-pairs $P --seed $SEED
      done
    done
  done
done
echo "done: $(date)"

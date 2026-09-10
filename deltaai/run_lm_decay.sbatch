#!/bin/bash
#SBATCH --job-name=smat-lm-decay
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/lm-decay-%j.out

# Does Mamba-2 style selective decay fix the dilution on enwik8?  T=4096,
# QK-norm, 20k steps, torch path (decayed scan).  References (no decay):
# d=1 2.98, d=2 2.44, d=2 no-LR 2.20, softmax 1.53.
# First a 3-minute MQAR check that the mask limit holds with decay (d=2).
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
echo "==== mqar d=2 decay mamba2   $(date) ===="
OMP_NUM_THREADS=4 python -u mqar_lm.py --arm smat --d 2 --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 \
    --qk-norm --short-conv 3 --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 \
    --layers 4 --d-model 256 --heads 4 --r 64 --steps 5000 --nb 32 --lr 1e-3 --eval-every 1000 --eval-n 256 \
    --out results/mqar_T512_gate_sweep.csv --tag _d2B
for ARM in "--arm smat --d 1" "--arm smat --d 2" "--arm smat --d 2 --no-lr"; do
    echo "==== arm: $ARM --decay mamba2   $(date) ===="
    OMP_NUM_THREADS=4 python -u train_lm.py $ARM --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm \
        --T 4096 --layers 4 --d-model 256 --heads 4 --r 64 \
        --steps 20000 --nb 8 --lr 3e-4 --warmup 500 --eval-every 4000 --val-windows 128 \
        --out results/lm_T4096_20k_qkn_decay.csv
done
echo "done: $(date)"

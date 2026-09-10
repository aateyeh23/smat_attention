#!/bin/bash
#SBATCH --job-name=smat-lm-20k
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/lm-20k-%j.out

# Same experiment as run_lm.sbatch, but all four arms in sequence on one GPU
# (the interactive partition allows one submission per user).
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"

for ARM in "--arm softmax" "--arm smat --d 1" "--arm smat --d 2" "--arm smat --d 3"; do
    echo "==== arm: $ARM   $(date) ===="
    OMP_NUM_THREADS=4 python -u train_lm.py $ARM \
        --T 4096 --layers 4 --d-model 256 --heads 4 --r 64 \
        --steps 20000 --nb 8 --lr 3e-4 --warmup 500 --eval-every 2000 --val-windows 256 \
        --out results/lm_T4096_20k.csv
done
echo "done: $(date)"

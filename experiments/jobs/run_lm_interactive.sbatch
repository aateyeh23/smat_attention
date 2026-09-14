#!/bin/bash
#SBATCH --job-name=smat-lm-seq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Same experiment as run_lm.sbatch, but all four arms in sequence on one GPU
# (the interactive partition allows one submission per user).
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"

for ARM in "--arm softmax" "--arm smat --d 1" "--arm smat --d 2" "--arm smat --d 3"; do
    echo "==== arm: $ARM   $(date) ===="
    OMP_NUM_THREADS=4 python -u src/smat/model.py $ARM \
        --T 4096 --layers 4 --d-model 256 --heads 4 --r 64 \
        --steps 3000 --nb 8 --lr 1e-3 --eval-every 500 --val-windows 128 \
        --out results/lm_T4096_seq.csv
done
echo "done: $(date)"

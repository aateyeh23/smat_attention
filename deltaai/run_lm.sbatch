#!/bin/bash
#SBATCH --job-name=smat-lm
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/lm-%A_%a.out

# Minimal NLP experiment: byte-level LM on enwik8, T=4096, 4 layers, one arm per
# array task, all on separate GPUs.  Results -> results/lm_T4096.csv (one row
# per eval per arm) and results/lm_T4096_<arm>.json.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"

ARMS=("--arm softmax" "--arm smat --d 1" "--arm smat --d 2" "--arm smat --d 3")
echo "==== arm: ${ARMS[$SLURM_ARRAY_TASK_ID]} ===="
OMP_NUM_THREADS=4 python -u train_lm.py ${ARMS[$SLURM_ARRAY_TASK_ID]} \
    --T 4096 --layers 4 --d-model 256 --heads 4 --r 64 \
    --steps 3000 --nb 8 --lr 1e-3 --eval-every 500 --val-windows 128 \
    --out results/lm_T4096.csv
echo "done: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-needle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Positional needle-in-a-haystack (tasks/needle.py), T=4096, enwik8 haystack.
# Task (d_ref, k): needles at the marked columns of the d_ref mask, query rows
# that isolate each needle under that mask.  Arms: softmax, smat d=1, smat d_ref.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
for TASK in "2 2" "3 3" "4 4"; do
    set -- $TASK; DREF=$1; K=$2
    for ARM in "--arm softmax" "--arm smat --d 1" "--arm smat --d $DREF"; do
        echo "==== task d_ref=$DREF k=$K   arm: $ARM   $(date) ===="
        OMP_NUM_THREADS=4 python -u tasks/needle.py $ARM --d-ref $DREF --k $K --qk-norm --triton-bwd \
            --T 4096 --layers 4 --d-model 256 --heads 4 --r 64 \
            --steps 3000 --nb 16 --lr 1e-3 --eval-every 500 --eval-n 512 \
            --out results/needle_T4096.csv
    done
done
echo "done: $(date)"

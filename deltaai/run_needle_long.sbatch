#!/bin/bash
#SBATCH --job-name=smat-needle-long
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=01:00:00
#SBATCH --output=logs/needle-long-%j.out

# Positional needle-in-a-haystack (needle_lm.py), T=4096, enwik8 haystack.
# Task (d_ref, k): needles at the marked columns of the d_ref mask, query rows
# that isolate each needle under that mask.  Arms: softmax, smat d=1, smat d_ref.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
# k=2 SMAT arms at 8000 steps (d=2 was still rising at 3000), then the k=4 task
for CFG in "2 2 1 8000" "2 2 2 8000" "4 4 1 3000" "4 4 4 3000"; do
    set -- $CFG; DREF=$1; K=$2; D=$3; STEPS=$4
    echo "==== task d_ref=$DREF k=$K   arm: --arm smat --d $D   steps=$STEPS   $(date) ===="
    OMP_NUM_THREADS=4 python -u needle_lm.py --arm smat --d $D --d-ref $DREF --k $K --qk-norm --triton-bwd \
        --T 4096 --layers 4 --d-model 256 --heads 4 --r 64 \
        --steps $STEPS --nb 16 --lr 1e-3 --warmup 400 --eval-every 1000 --eval-n 512 \
        --out results/needle_T4096_long.csv
done
echo "done: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-aux
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out

# The secondary sweeps of run_smat.sbatch, split off so the main bf16 sweep is
# not competing with them for the wall clock.  Writes disjoint outputs, so the
# two can be submitted together.
set -uo pipefail
source experiments/prelude.sh

echo "================ sweep: fp32 ================"
python -u analysis/bench_prefill.py --device cuda --dtype fp32 \
    --T 4096 8192 16384 32768 65536 131072 \
    --d 1 2 3 4 \
    --nb 8 --r 64 --d-qk 64 --d-v 64 --chunk 128 \
    --max-I-tok 20000000 --max-I-prof-torch 600000 \
    --repeat 10 --warmup 3 --out results/results_fp32.csv

echo "======== constant sensitivity: the feature rank r, at d = 3 ========"
for R in 32 64 128 256; do
    python -u analysis/bench_prefill.py --device cuda --dtype bf16 \
        --T 4096 8192 16384 32768 65536 131072 \
        --d 3 --nb 8 --r "$R" --d-qk 64 --d-v 64 --chunk 128 \
        --no-direct --no-subtractive --repeat 10 --warmup 3 \
        --out "results/results_r${R}.csv"
done

echo "======== chunk width c, at d = 3 ========"
for C in 64 128 256; do
    python -u analysis/bench_prefill.py --device cuda --dtype bf16 \
        --T 16384 65536 262144 \
        --d 3 --nb 8 --r 64 --d-qk 64 --d-v 64 --chunk "$C" \
        --no-direct --no-subtractive --repeat 10 --warmup 3 \
        --out "results/results_c${C}.csv"
done
echo "done: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-routing
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-routing-%j.out

# The expressivity benchmark of Sec. 4, both halves.
#
# The certified half (bench_routing.py) needs no GPU and is cheap, but it is run
# here anyway so the ceiling in routing_T1024.csv and the learned curve in
# routing_learned.csv come from the same marked sets -- choose_marked_set is
# seeded, and a mismatched pair would silently compare a learned run against the
# wrong ceiling.
#
# Training must use the torch backends: the Triton kernels are not
# autograd.Functions, so "auto" would select them on CUDA and gradients would be
# wrong or missing without an error.  train_routing.py hard-sets the backend and
# asserts a finite gradient before every mask it trains on.

set -euo pipefail
source "$(dirname "$0")/_prelude.sh"

echo "================ certified ceiling ================"
python -u bench_routing.py \
    --T 1024 --d 1 2 3 4 5 --kmax 6 \
    --dtype fp64 --seed 0 \
    --out results/routing_T1024.csv

echo
echo "================ learned run ================"
# 4 arms x 6 k x 3 seeds; softmax arms are mask-free so they run once, not once
# per d.  1500 steps of batch 64 is ~100k examples per run against 2^k <= 64
# distinct requests, with payloads resampled every batch.
python -u train_routing.py \
    --T 1024 --d 1 2 3 4 5 --kmax 6 \
    --arms smat smat_content softmax softmax_content \
    --steps 1500 --nb 64 --lr 3e-3 --seeds 3 \
    --device cuda \
    --out results/routing_learned.csv

echo
echo "================ figures ================"
# Draws fig12 and fig13 together, pairing the learned run with the T=1024
# ceiling written above.
python plot_smat.py --results-dir results --outdir figs
python plot_smat.py --results-dir results --outdir figs_dark --dark

echo "done: $(date)"

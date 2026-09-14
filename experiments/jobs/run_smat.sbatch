#!/bin/bash
#SBATCH --job-name=smat-bench
#SBATCH --gres=gpu:1                # H200 also available on this partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --output=logs/%x-%j.out

# SMAT timing sweep for the revised Section 3.
#
# Correctness first -- a subtly wrong mask produces plausible-but-meaningless
# timings -- then the T sweeps.  The exact VC search is CPU-only and slow, so it
# lives in analysis/verify_theory.py and is not run here; run that on a login node.

set -euo pipefail
source experiments/prelude.sh

echo "================ correctness (fp64 on cpu, kernels on gpu) ================"
# Do not abort the sweep on a kernel failure: smat_attn falls back to the torch
# path with a warning, and a fallback timing is still worth having.  But say so
# loudly, because a wrong mask makes every number below meaningless.
if ! OMP_NUM_THREADS=4 python -u test_smat.py --skip-vc; then
    echo
    echo "!!!!!!!! CORRECTNESS CHECKS FAILED -- read the block above before"
    echo "!!!!!!!! trusting any timing in this log."
fi

echo
echo "================ sweep: bf16 GEMM / fp32 accumulate ================"
python -u analysis/bench_prefill.py \
    --device cuda --dtype bf16 \
    --T 4096 8192 16384 32768 65536 131072 262144 \
    --d 1 2 3 4 \
    --nb 8 --r 64 --d-qk 64 --d-v 64 --chunk 128 \
    --max-I-tok 20000000 \
    --repeat 15 --warmup 5 \
    --out results/results_bf16.csv

# The fp32, r and c sweeps live in run_smat_aux.sbatch so that they do not
# compete with the bf16 sweep for the wall clock; submit both.

echo
echo "================ figures ================"
# plot_smat draws every panel whose CSV is present, so this picks up the aux
# sweeps and the theory and routing tables too if those jobs have already run.
python analysis/figures.py results/results_bf16.csv --results-dir results --outdir figs
python analysis/figures.py results/results_bf16.csv --results-dir results \
    --outdir figs_dark --dark

echo "done: $(date)"

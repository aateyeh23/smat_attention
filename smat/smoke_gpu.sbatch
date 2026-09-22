#!/bin/bash
#SBATCH --job-name=smat-smoke
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:35:00
#SBATCH --output=logs/smoke-%j.out

# A short version of run_smat.sbatch: the Triton kernels get a real GPU and a
# real sweep, but at a wall clock that schedules in minutes.  Run this before
# the 3.5-hour job, not instead of it.
set -uo pipefail
source "$(dirname "$0")/_prelude.sh"

echo "==== correctness incl. triton kernels ===="
OMP_NUM_THREADS=4 python -u test_smat.py --skip-vc

echo "==== tiny gpu sweep ===="
python -u bench_smat.py --device cuda --dtype bf16 \
    --T 4096 8192 16384 32768 --d 1 2 3 4 \
    --nb 8 --r 64 --d-qk 64 --d-v 64 --chunk 128 \
    --repeat 8 --warmup 3 --out /tmp/smoke.csv
echo "done: $(date)"

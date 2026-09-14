#!/bin/bash
#SBATCH --job-name=smat-benchca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# The last red note in Section 4: the PREFILL schedule under a content-addressed
# assignment has never been timed.  Pooling by profile is a dense contraction,
# one pass over the distant block per cell, Theta(N0 n r p) where a grouped
# reduction is O(n r p); the read builds every (direction, offset) state for
# every recent row and then selects, Theta(T_R B r p) where Theta(T_R r p)
# suffices.  Values agree either way -- the forward weights are one-hot -- so the
# accuracy numbers are unaffected, which is exactly why only the timing is open.
#
# analysis/bench_assign.py implements the scatter by profile and the counting sort of queries
# by type that the note names, checks both against the dense reference for
# exactness, and times them.  On CPU at small shapes the pool is 3-5x with
# agreement to 6e-06, which is fp32 rounding.
set -uo pipefail
source experiments/prelude.sh

for D in 2 3; do
  echo "==== d=$D   $(date) ===="
  OMP_NUM_THREADS=8 python -u analysis/bench_assign.py --d $D --nb 2 --heads 4 --r 64 --dv 64 \
    --T 1024 4096 16384 65536 262144 --dense-max-T 16384 --repeat 5 \
    --out results/bench_ca.csv || echo "FAILED: d=$D"
done
echo "BENCHCA DONE: $(date)"

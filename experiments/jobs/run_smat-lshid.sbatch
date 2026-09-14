#!/bin/bash
#SBATCH --job-name=smat-lshid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# the LSH bucketing baseline with the id hash.
#
# Every multi-item number so far used a frozen random projection of the token
# EMBEDDING, whose hit rate is 0.86-0.96 and whose row density drifts upward
# during training because learned embeddings can move a token between cells.
# Hashing the token id directly removes both: on single-item recall it gives hit
# rate exactly 1.000, density pinned at 1/N0 with no drift across any seed or
# layer, accuracy 0.999-1.000 against the embedding hash's 0.934-0.976, and
# convergence in 1000 steps rather than 4000.
#
# So the staircase deserves rerunning with it.  Hash quality was a plausible
# cause of the noise in the cells that should succeed (d=3, k=2 read
# 0.955 / 0.654 / 0.613 across seeds); with a perfect hash the k <= d-1 pattern
# should be cleaner, and any residual failure is the geometry rather than the
# assignment.  Both arms of the comparison get the same hash, so the only
# difference remains whether a query addresses a union of cells or one.
set -uo pipefail
source experiments/prelude.sh

MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 8 --lr 1e-3 --steps 4000 --eval-every 1000 --dirs oracle \
  --hash-src id --read-mode point --out results/mkar_lsh_id.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py $MK "${@:2}" || echo "FAILED: $1"; }

for D in 3 4; do
  for K in 1 2 3; do
    run "d$D k$K" --d $D --k $K --seed 0
  done
done
echo "DONE: $(date)"

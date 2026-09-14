#!/bin/bash
#SBATCH --job-name=smat-lsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# LSH-bucketing baseline for multi-item recall.
#
# Hashing tokens into buckets and attending within a bucket is the mechanism of
# LSH attention; in this framework it is exactly a query reading its own cell
# and nothing else, since distinct profiles occupy DISJOINT cells.  Everything
# else is held fixed -- the same frozen content hash, the same offset, the same
# recurrent state per bucket, the same single layer -- so the only difference is
# whether a query may address a structured union of cells (a hyperplane) or just
# one.  That isolates the geometry from the hashing.
#
# At d=2 the two coincide: a hyperplane of F_q^1 IS a single cell, so our d=2
# row already is the baseline.  The comparison lives at d=3 and d=4.
#
# Prediction: the baseline solves k=1 at every d and nothing above it, because
# disjoint cells cannot cover two requested profiles at once.  Ours: k <= d-1.
set -uo pipefail
source experiments/prelude.sh

MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 8 --lr 1e-3 --steps 4000 --eval-every 1000 --dirs oracle \
  --read-mode point --out results/mkar_lsh.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py $MK "${@:2}" || echo "FAILED: $1"; }

for D in 3 4; do
  for K in 1 2 3; do
    run "lsh d$D k$K" --d $D --k $K --seed 0
  done
done
echo "LSH DONE: $(date)"

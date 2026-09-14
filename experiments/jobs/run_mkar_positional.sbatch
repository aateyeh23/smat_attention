#!/bin/bash
#SBATCH --job-name=smat-mkarpos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# The missing cell of (task x assignment).  Multi-item recall has only ever been
# run with a content-addressed assignment; k-subset routing only with a
# positional one.  This runs multi-item recall POSITIONALLY, on the diagonal
# cells the staircase rests on.
#
# Expected: at chance.  The pairs sit at random positions, so a map from query
# position to hyperplane carries no information about where the requested
# payloads are, exactly as in the single-item case (positional 0.012 against
# content 0.976).  If it is at chance, the two axes are cleanly separated: d
# decides what subsets a row can form, the assignment decides whether the row
# lands on the right ones.
set -uo pipefail
source experiments/prelude.sh

MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 8 --lr 1e-3 --steps 4000 --eval-every 1000 --dirs oracle \
  --out results/mkar_positional.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py $MK "${@:2}" || echo "FAILED: $1"; }

for S in 0 1; do
  run "pos d2 k1 s$S" --assign positional --d 2 --k 1 --seed $S
  run "pos d3 k2 s$S" --assign positional --d 3 --k 2 --seed $S
  run "pos d4 k3 s$S" --assign positional --d 4 --k 3 --seed $S
done
echo "MKARPOS DONE: $(date)"

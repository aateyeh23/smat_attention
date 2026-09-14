#!/bin/bash
#SBATCH --job-name=smat-mkbase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Real baselines on multi-item recall.
#
# The staircase so far is compared only against our own bucketing ablation, so
# it demonstrates that the geometry matters within our construction and says
# nothing about whether the construction is worth having.  These are the two
# references that decide that:
#
#   softmax   no mask at all.  Every distant payload is visible and the kernel
#             alone has to isolate the k requested ones.  This is the ceiling: if
#             softmax solves k=3 easily, the task is not hard and the staircase
#             is a statement about our mask rather than about the problem.
#   d=1       linear attention, i.e. the all-ones long-range block.  Same
#             visibility as softmax but a rank-r state instead of a score matrix,
#             so it is the efficient-attention reference the paper must beat.
#
# Every arm gets a width-3 causal conv.  Without it a single layer cannot form an
# induction head -- from the key token at the answer row it must find where that
# key sits and then read the NEXT position -- and softmax scored 0.000 at k=1 in
# the first attempt for exactly that reason.  Our arms get that composition free
# through prof(j) = hash(token j-1), so withholding it from the baselines is the
# comparison answering a question about layer count rather than about the mask.
#
# Both at k = 1, 2, 3 and two seeds, single layer, matched to the staircase runs
# (T=1024, N=8 pairs, 4000 steps).  Existing rows to compare against, seed 0:
#   ours d=3 plane: 0.711 / 0.955 / 0.143
#   bucketing d=3:  0.998 / 0.000 / 0.000
set -uo pipefail
source experiments/prelude.sh

MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 8 --lr 1e-3 --steps 4000 --eval-every 1000 --hash-src id --short-conv 3 \
  --out results/mkar_baselines.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py $MK "${@:2}" || echo "FAILED: $1"; }

for S in 0 1; do
  for K in 1 2 3; do
    run "softmax k$K s$S"    --arm softmax --d 2 --k $K --seed $S
    run "d1 linear k$K s$S"  --arm smat --d 1 --k $K --seed $S
  done
done
echo "BASELINES DONE: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-seeds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# Seeds for the two single-seed results the paper now leans on.  This project
# has produced two seed artefacts already -- a d=3 hyperplane arm read 0.450 on
# one seed and 0.835 / 0.855 on two others, and the d=2-over-d=3 margin vanished
# once averaged -- so neither result should go in on one draw.
#
# Part 1, the staircase (Table "mkar").  Seeds 1 and 2 of the diagonal and the
# cells immediately above it, which are what the k <= d-1 claim rests on:
#   d=2 k=2, d=3 k=2, d=3 k=3, d=4 k=3.
# The k=1 column is not repeated; it is solvable at every d and carries no
# weight in the claim.
#
# Part 2, the curriculum (0.911, and 0.952 with the agreement term).  Seeds 1
# and 2 of both, against a frozen reference of 0.940-0.976 measured on three.
set -uo pipefail
source experiments/prelude.sh

MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 8 --lr 1e-3 --steps 4000 --eval-every 1000 --dirs oracle \
  --out results/mkar_seeds.csv"
AL="--arm smat --d 2 --assign content --read-mode point --hash-shift 1 \
  --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --steps 3000 \
  --eval-every 500 --eval-n 256 --early-stop 0.995 --balance-coef 0.01 \
  --hash-src embed --hash-freeze --unfreeze-at 1500 \
  --out results/mqar_align_seeds.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u "${@:2}" || echo "FAILED: $1"; }

for S in 1 2; do
  run "mkar d2 k2 s$S" tasks/mkar.py $MK --d 2 --k 2 --seed $S
  run "mkar d3 k2 s$S" tasks/mkar.py $MK --d 3 --k 2 --seed $S
  run "mkar d3 k3 s$S" tasks/mkar.py $MK --d 3 --k 3 --seed $S
  run "mkar d4 k3 s$S" tasks/mkar.py $MK --d 4 --k 3 --seed $S
done
for S in 1 2; do
  run "curriculum s$S"             tasks/mqar.py $AL --seed $S
  run "curriculum+agreement s$S"   tasks/mqar.py $AL --agree-coef 1.0 --seed $S
done
echo "SEEDS DONE: $(date)"

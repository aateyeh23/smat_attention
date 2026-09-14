#!/bin/bash
#SBATCH --job-name=smat-fair2
#SBATCH --partition=jsteinhardt,songmei,berkeleynlp,yugroup
#SBATCH --exclude=morgoth,merry,roo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/fair2-%j.out

# Baselines at the budget that is now known to work, on BOTH set-valued tasks.
#
# Softmax on multi-item recall has a phase transition whose onset depends on lr
# AND batch size: at lr 3e-4 with batch 32 it goes 0.000 -> 1.000 between steps
# 4000 and 8000; at lr 3e-4 with batch 8 it is still 0.000 at 16000; at lr 1e-3
# it never transitions at any budget tried.  Three earlier baseline runs sat on
# the wrong side of that surface, each time producing an apparent failure that
# flattered us.  Our own arms converge by step 1000 at batch 8 because the mask
# removes the search, so any budget tuned to our convergence is far too small for
# a model that has to find the mechanism.  Baselines therefore get 4x the batch
# and 3x the steps here, which is the conservative direction.
#
# Both tasks, because the counting result has the same exposure: its d=1 arm ran
# at lr 1e-3, batch 8, 4000 steps and scored 0.32 at k=1, which cannot be
# distinguished from undertraining without this.
#
# Reference numbers to beat, all at batch 8 / 4000 steps:
#   recall   ours d=3 k=2 0.979,  d=4 k=3 0.986
#   count    ours d=3 k=2 0.922,  d=4 k=3 0.762,  d=2 k=1 0.986
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

COMMON="--T 1024 --chunk 256 --n-ans 8 --layers 1 --d-model 256 --heads 4 --r 64 \
  --nb 32 --lr 3e-4 --steps 12000 --eval-every 2000 --seed 0 --short-conv 3"
rec() { echo "==== recall $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mkar_lm.py $COMMON \
  --n-pairs 8 --hash-src id --out results/fair2_recall.csv "${@:2}" || echo "FAILED: $1"; }
cnt() { echo "==== count $1   $(date) ===="; OMP_NUM_THREADS=8 python -u count_lm.py $COMMON \
  --n-marks 12 --n-syms 8 --out results/fair2_count.csv "${@:2}" || echo "FAILED: $1"; }

for K in 1 2 3; do
  rec "softmax k$K" --arm softmax --d 2 --k $K
  rec "d1 k$K"      --arm smat --d 1 --k $K
  cnt "softmax k$K" --arm softmax --d 2 --k $K
  cnt "d1 k$K"      --arm smat --d 1 --k $K
done
echo "FAIR2 DONE: $(date)"

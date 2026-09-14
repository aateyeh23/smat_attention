#!/bin/bash
#SBATCH --job-name=smat-fairbase
#SBATCH --partition=jsteinhardt,songmei,berkeleynlp,yugroup
#SBATCH --exclude=morgoth,merry,roo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mkar-fairbase-%j.out

# Baselines on multi-item recall at a budget that lets them learn.
#
# Softmax has a sharp phase transition on this task: at lr 3e-4 it sits at
# exact_support 0.000 with loss pinned at 0.376 through step 4000, then reaches
# 1.000 by step 8000.  At lr 1e-3 it never escapes, even at 20k steps.  Every
# baseline number collected so far used lr 1e-3 and 4000 steps -- on the wrong
# side of that transition in both respects -- so all of them are artefacts.
#
# This is the decisive run for the multi-item claim.  Our arms reach 0.979 at
# d=3,k=2 and 0.986 at d=4,k=3 with a mask that provably admits the requested
# set.  If softmax, given a fair budget, also solves k=2 and k=3, then the
# staircase describes our construction and not the difficulty of the problem,
# and the multi-item result is a statement about masks rather than a capability
# advantage.  If softmax solves k=1 but stalls above it, the task is hard for a
# reason the geometry addresses.
#
# lr 3e-4, 16000 steps, k = 1, 2, 3, softmax and linear attention.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 8 --lr 3e-4 --steps 16000 --eval-every 2000 --hash-src id --short-conv 3 \
  --seed 0 --out results/mkar_fairbase_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mkar_lm.py $MK "${@:2}" || echo "FAILED: $1"; }

for K in 1 2 3; do
  run "softmax k$K"   --arm softmax --d 2 --k $K
  run "d1 linear k$K" --arm smat --d 1 --k $K
done
echo "FAIRBASE DONE: $(date)"

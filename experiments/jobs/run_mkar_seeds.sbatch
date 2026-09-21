#!/bin/bash
#SBATCH --job-name=smat-mkar-seeds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out

# Seed repeats for the multi-key subset recall table.
#
# Every fair2 run so far is seed 0, so Table 5 is n = 1 and has no spread to
# report.  This repeats the rows the table actually prints -- softmax, d = 1,
# the plane reads at d = 2, 3, 4 and the single-cell reads at d = 3, 4, at
# k = 1, 2, 3 -- under seeds 1 and 2.  The protocol is copied verbatim from
# run_mkar_fair2.sbatch (batch 32, lr 3e-4, 12000 steps, T = 1024, 8 pairs,
# one layer, id hash, short conv 3); only --seed and --out differ, and the run
# name carries the seed, so analysis/table_mkar.py averages them by label.
#
#   sbatch run_mkar_seeds.sbatch 1 base     # softmax and d=1, k = 1,2,3
#   sbatch run_mkar_seeds.sbatch 1 smat23   # d = 2, 3 plane oracle
#   sbatch run_mkar_seeds.sbatch 1 smat4    # d = 4 plane oracle
#   sbatch run_mkar_seeds.sbatch 1 point    # d = 3, 4 single cell
#   sbatch run_mkar_seeds.sbatch 0 finish   # the seed-0 runs that were cut short
#
# "finish" re-runs the three fair2 runs that stopped before 12000 steps: the
# d=3 k=3 oracle cell, which the main table prints, and two diagnostic heads.
# Their rows carry the same run names as the truncated ones, and table_mkar.py
# keeps the largest step per run, so the short rows are superseded rather than
# averaged in.
set -uo pipefail
source experiments/prelude.sh

SEED="${1:-1}"
GROUP="${2:-base}"
MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 32 --lr 3e-4 --steps 12000 --eval-every 2000 --hash-src id --short-conv 3 \
  --seed ${SEED} --tag _fair2 --out results/mkar_fair2_s${SEED}_${GROUP}.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py $MK "${@:2}" || echo "FAILED: $1"; }

case "$GROUP" in
  base)
    for K in 1 2 3; do
      run "softmax k$K"   --arm softmax --d 2 --k $K
      run "d1 linear k$K" --arm smat --d 1 --k $K
    done ;;
  smat23)
    for D in 2 3; do for K in 1 2 3; do
      run "d$D k$K" --arm smat --d $D --k $K --read-mode plane --dirs oracle
    done; done ;;
  smat4)
    for K in 1 2 3; do
      run "d4 k$K" --arm smat --d 4 --k $K --read-mode plane --dirs oracle
    done ;;
  point)
    for D in 3 4; do for K in 1 2 3; do
      run "point d$D k$K" --arm smat --d $D --k $K --read-mode point --dirs oracle
    done; done ;;
  finish)
    run "d3 k3 oracle"   --arm smat --d 3 --k 3 --read-mode plane --dirs oracle
    run "fixed d3 k3"    --arm smat --d 3 --k 3 --dirs learned --dir-head fixed --dir-window 2
    run "mlp d4 k3"      --arm smat --d 4 --k 3 --read-mode plane --dirs learned --dir-head mlp ;;
esac
echo "MKAR SEEDS ${SEED} ${GROUP} DONE: $(date)"

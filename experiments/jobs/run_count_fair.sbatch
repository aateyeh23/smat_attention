#!/bin/bash
#SBATCH --job-name=smat-cntfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Our counting arms at the budget the baselines got.
#
# The counting staircase was measured at batch 8 / lr 1e-3 / 4000 steps, where
# the no-mask d=1 arm scored 0.324 / 0.268 / 0.238.  At batch 32 / lr 3e-4 /
# 12000 steps the same arm scores 0.990 at k=1 and 0.998 at k=2 -- so that row
# was undertraining, and with it the claim that the mask is needed for
# aggregation.  Our own arms have not been run at the larger budget either, so
# the table currently compares a tuned baseline against an untuned treatment,
# which is the same error as before with the sign flipped.
#
# Expectation: d=2 should stay low at k=2 whatever the budget, since one cell
# cannot cover two content-defined groups and the model is single-layer with no
# route around the mask.  If it climbs anyway, something is reaching the second
# group that the geometry says should be invisible, and that needs finding.
set -uo pipefail
source experiments/prelude.sh

C="--T 1024 --chunk 256 --n-marks 12 --n-syms 8 --n-ans 8 --layers 1 --d-model 256 \
  --heads 4 --r 64 --nb 32 --lr 3e-4 --steps 12000 --eval-every 2000 --seed 0 \
  --out results/count_fair.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/counting.py $C "${@:2}" || echo "FAILED: $1"; }

for D in 2 3 4; do
  for K in 1 2 3; do
    run "d$D k$K" --arm smat --d $D --k $K
  done
done
echo "COUNTFAIR DONE: $(date)"

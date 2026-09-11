#!/bin/bash
#SBATCH --job-name=smat-mkarleak
#SBATCH --partition=jsteinhardt,songmei,berkeleynlp,yugroup
#SBATCH --exclude=morgoth,merry,roo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:15:00
#SBATCH --output=logs/mkar-leak-%j.out

# PRIORITY 1: find the leak in multi-key subset recall.
#
# d=2 k=2 scores exact_support 1.000 with ONE layer, yet the mask admits only
# 53% of the requested payloads (measured: the query's own key always, the other
# only on a 1/23 cell collision).  So the model answers correctly about half the
# time it cannot see the evidence.  Until that channel is found, every number
# this task produces is untrustworthy, and it is the only task that can tie d to
# a content-addressed problem.
#
#   none     reproduce the anomaly
#   nolr     long-range block disabled entirely.  If this still scores ~1.0 the
#            mask is irrelevant and the leak is in the task, not the geometry.
#   permidx  a pair's index is a random permutation rather than its position
#            rank, in case rank is recoverable some other way
#   shuftgt  targets unrelated to the request; anything above chance indicts the
#            readout or the metric
#
# d=3 k=2 is included because it is a cell the geometry says SHOULD work, so it
# separates "the task is broken everywhere" from "only the impossible cells leak".
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

BASE="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 8 --lr 1e-3 --steps 2000 --eval-every 500 --seed 0 --dirs oracle \
  --out results/mkar_leak_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mkar_lm.py $BASE "${@:2}" || echo "FAILED: $1"; }

for A in none nolr permidx shuftgt; do
  run "d2 k2 $A" --d 2 --k 2 --ablate $A
done
run "d3 k2 none" --d 3 --k 2 --ablate none
run "d3 k2 nolr" --d 3 --k 2 --ablate nolr
echo "LEAK DONE: $(date)"

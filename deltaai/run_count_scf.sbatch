#!/bin/bash
#SBATCH --job-name=smat-count
#SBATCH --partition=jsteinhardt,songmei,berkeleynlp,yugroup
#SBATCH --exclude=morgoth,merry,roo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/count-%j.out

# Multi-group counting: does the k <= d-1 staircase survive when the task is
# AGGREGATION rather than retrieval?
#
# Multi-item recall showed d binding on a task that retrieves a set.  The claim
# ought to be broader -- d is the number of context items a layer can select
# independently, so it should bind wherever one position computes a function of
# a union of k content-defined groups, whatever that function is.  Counting
# shares no machinery with retrieval: no payload identity, no support readout,
# a single integer answer.  If the staircase appears here too, the condition is
# about set-valued queries rather than about recall, and that is the general
# statement the paper wants.  If it does not, the recall result is narrower than
# we have been claiming.
#
# 12 countable tokens at random positions among 8 symbols; a query names k
# symbols and must emit how many of the 12 carry any of them.  hash_shift is 0:
# a countable token is filed under its own symbol, unlike recall where a payload
# had to be filed under the key preceding it.
#
# d arms first, per priority; softmax follows in a separate job.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

C="--T 1024 --chunk 256 --n-marks 12 --n-syms 8 --n-ans 8 --layers 1 --d-model 256 \
  --heads 4 --r 64 --nb 8 --lr 1e-3 --steps 4000 --eval-every 1000 --seed 0 \
  --out results/counting_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u count_lm.py $C "${@:2}" || echo "FAILED: $1"; }

for D in 1 2 3 4; do
  for K in 1 2 3; do
    run "d$D k$K" --arm smat --d $D --k $K
  done
done
echo "COUNT DONE: $(date)"

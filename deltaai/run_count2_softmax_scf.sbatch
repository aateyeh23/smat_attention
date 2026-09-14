#!/bin/bash
#SBATCH --job-name=smat-cnt2sm
#SBATCH --partition=jsteinhardt,songmei,berkeleynlp,yugroup
#SBATCH --exclude=morgoth,merry,roo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/count2sm-%j.out

# Counting, rebuilt.  The first version put the symbol ON the counted token and
# drew from 8 symbols against rank r=64, so a query could hold the requested set
# as a sum of symbol features and phi_q . psi_j was already the membership
# indicator: the no-mask d=1 arm then scored 0.990 / 0.998 at the fair budget and
# the mask was a handicap, not a help.  Two changes remove that route.
#
#   1. The symbol sits at p and the counted MARK at p+1, so group membership is
#      only available by composing with the preceding token, as in recall.
#   2. 512 symbols with a fresh 8-symbol pool per sequence, so no fixed set of
#      near-orthogonal per-symbol features fits in rank 64.
#
# Both changes make the task depend on LOCAL composition, and the first pass of
# this grid was run with --short-conv 1, which is a depthwise kernel of size 1:
# a per-channel scale with no neighbour mixing at all.  In a one-layer model the
# answer row then saw only its QMARK -- the k requested symbols sit at t-k..t-1
# -- and the counted MARK never saw its symbol.  Every arm without oracle
# directions sat at the marginal-count floor, and smat d=1 and softmax produced
# byte-identical curves (0.338 / 0.249 / 0.229 at k=1,2,3), which is the tell.
# Width 4 is the smallest that reaches t-3, so it covers the request for k up to
# 3 and the symbol/mark pair at p, p+1.  Archived floor runs: results/_conv1/.
#
# The geometry is verified directly on the data, not inferred: the fraction of
# requested counted items admitted by the answer row's hyperplane is exactly
# 1.000 where k <= d-1 and 0.40-0.53 where it is not.  Admission also lets in
# 0.3-2.1 unrequested items per row, which the kernel must reject on content --
# that is what the short conv is for, and it is on for every arm here.
#
# So the prediction is the staircase again, now with d=1 in it: d=1 flat and low
# at every k, d=2 only at k=1, d=3 up to k=2, d=4 up to k=3.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

C="--T 1024 --chunk 256 --n-marks 12 --n-syms 512 --pool 8 --n-ans 8 --layers 1 \
  --d-model 256 --heads 4 --r 64 --nb 32 --lr 3e-4 --steps 12000 --eval-every 2000 \
  --short-conv 4 --seed 0 --out results/count2_softmax_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u count_lm.py $C "${@:2}" || echo "FAILED: $1"; }

for K in 1 2 3; do
  run "softmax k$K" --arm softmax --d 4 --k $K
done
echo "COUNT2SM DONE: $(date)"

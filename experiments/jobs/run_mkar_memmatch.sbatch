#!/bin/bash
#SBATCH --job-name=smat-mkar-memmatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Memory-matched linear attention on multi-key subset recall.
#
# The d=1 row answers "does the mask matter?" only if the reader accepts that
# the two arms hold the same amount; d=1 keeps one state of r x (d_head + 1)
# and SMAT keeps one per profile.  This gives linear attention the whole budget
# by widening its feature map instead: r' = N0 * r, the same accounting
# (state_per_head = r * (d_head + 1)) as the MQAR memory-matched runs, which
# used r = 1472 for exactly this reason.
#
#   T = 1024, chunk 256, r = 64:  N0 = 23 (d=2), 121 (d=3), 125 (d=4)
#   so r' = 1472, 7744, 8000.
#
# Only d=2's budget is runnable: at r' = 7744 the feature tensors alone are
# ~4 GB per head at this batch.  So the matched arm carries 23x the default
# linear attention's memory, which covers the d=2 row outright and is a lower
# bound for d=3 and d=4 -- and the fact that those cannot be matched at all is
# worth stating rather than hiding.
#
#   sbatch run_mkar_memmatch.sbatch 1472
set -uo pipefail
source experiments/prelude.sh

R="${1:-1472}"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py "${@:2}" || echo "FAILED: $1"; }
for SEED in 0 1 2; do for K in 1 2 3; do
  run "memmatch r$R s$SEED k$K" --T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 \
      --d-model 256 --heads 4 --r $R --nb 32 --lr 3e-4 --steps 12000 \
      --eval-every 2000 --hash-src id --short-conv 3 --seed $SEED \
      --tag "_fair2_mem${R}" --out results/mkar_fair2_mem${R}.csv \
      --arm smat --d 1 --k $K
done; done
echo "MKAR MEMMATCH r$R DONE: $(date)"

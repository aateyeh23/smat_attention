#!/bin/bash
#SBATCH --job-name=smat-mkar-softsweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# Does softmax reliably solve multi-key subset recall, given an optimiser that
# lets it train?
#
# Under fair2 (constant lr 3e-4, no warmup) softmax solved 5 of 9 seed x k
# runs; the other 4 sat on their initial plateau for all 12000 steps (seed 1,
# k=2: loss 0.563 at every evaluation), including two failures at k=1.  That
# is an optimisation failure, not a capacity result, and the table cannot report
# a baseline that trains on half its seeds.  This sweeps the optimiser only:
#
#   sbatch run_mkar_softmax_sweep.sbatch wu_lr3e-4   # warmup alone
#   sbatch run_mkar_softmax_sweep.sbatch wu_lr1e-3   # warmup, higher lr
#   sbatch run_mkar_softmax_sweep.sbatch wu_lr1e-4   # warmup, lower lr
#
# Every config runs all three seeds at every k, the passing ones included, so a
# config is judged on how many of 9 it solves rather than on the four it fixes.
# The rest of the protocol is fair2 verbatim.  Output goes to mkar_softsweep_*,
# outside the mkar_fair2_* glob that table_meanstd.py and mkar_sanity.py read: these are a baseline
# check, and a config chosen from them would be rerun, not quoted from here.
set -uo pipefail
source experiments/prelude.sh

CFG="${1:-wu_lr3e-4}"
case "$CFG" in
  wu_lr3e-4) LR=3e-4 ;;
  wu_lr1e-3) LR=1e-3 ;;
  wu_lr1e-4) LR=1e-4 ;;
  *) echo "unknown config $CFG" >&2; exit 1 ;;
esac
WU=1000

run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py "${@:2}" || echo "FAILED: $1"; }
for SEED in 0 1 2; do for K in 1 2 3; do
  run "softmax s$SEED k$K $CFG" --T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 \
      --d-model 256 --heads 4 --r 64 --nb 32 --lr $LR --warmup $WU --steps 12000 \
      --eval-every 2000 --hash-src id --short-conv 3 --seed $SEED \
      --tag "_fair2_${CFG}" --out results/mkar_softsweep_${CFG}.csv \
      --arm softmax --d 2 --k $K
done; done
echo "SOFTMAX SWEEP $CFG DONE: $(date)"

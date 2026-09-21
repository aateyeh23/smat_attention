#!/bin/bash
#SBATCH --job-name=smat-mkar-posctrl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# Positive control for the multi-key table.
#
# At T=1024 every VC-1 baseline -- linear attention, memory-matched linear
# attention, Mamba-2, DeltaNet, Gated DeltaNet, log-linear -- sits at the
# base-rate loss for all 12000 steps, at every k, k=1 included.  k=1 is within
# what a VC-1 mask can route (every one of them solves k=1 on d-Subset Routing),
# so the zeros are not the VC ceiling.  The candidate explanation is noise: ~1000
# random tokens are written into one fixed-size state, and a model has to learn
# to stop writing them before it gets any gradient toward the task.
#
# This runs the same harness, same block, same protocol, at k=1 on shorter
# contexts, where there is far less to write:
#
#   T=128  ~110 noise tokens      T=256  ~240 noise tokens
#
# If the recurrent arms learn here, the harness trains them and the T=1024 zeros
# are a property of that setting, which the paper can then say.  If they fail
# even at T=128, something in the harness is wrong for them.  One seed: this is
# a diagnostic, not a table.
set -uo pipefail
source experiments/prelude.sh

run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py "${@:2}" || echo "FAILED: $1"; }
for T in 128 256; do
  C=$(( T / 4 ))
  COMMON="--T $T --chunk $C --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
    --r 64 --nb 32 --lr 3e-4 --steps 12000 --eval-every 2000 --hash-src id --short-conv 3 \
    --seed 0 --k 1 --tag _posctrl --out results/mkar_posctrl.csv"
  run "softmax T$T"        $COMMON --arm softmax --d 2
  run "linear T$T"         $COMMON --arm smat --d 1
  for A in mamba2 deltanet gated_deltanet loglinear; do
    run "$A T$T"           $COMMON --arm $A --d 2
  done
done
echo "POSCTRL DONE: $(date)"

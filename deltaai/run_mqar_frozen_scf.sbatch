#!/bin/bash
#SBATCH --job-name=smat-mqar-frz
#SBATCH --partition=yugroup,jsteinhardt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mqar-frz-%j.out

# Does the mask/read machinery work when the hash is right?  Take learning out of
# the hash: a frozen random projection of the TOKEN EMBEDDING, which is identical
# for a repeated token, so there is no invariance to learn and no way to collapse.
#
# The shift is the other half.  MQAR's payload sits one position after its key, so
# the state a query must see is the payload's; its profile therefore has to be the
# hash of the PRECEDING token.  Hashing each position's own token puts the payload
# in the wrong cell no matter how good the hash is -- which on its own would
# produce the 0.01 accuracies we have been reading as a hash failure.
#
#   A  frozen embedding hash + shift 1   <- both fixes; the decisive arm
#   B  frozen embedding hash, no shift   <- isolates the shift
#   C  learned hidden-state hash + shift <- does the shift alone rescue learning?
#
# d = 2 point mode only: d = 3 says nothing until the hash agrees with itself.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

COMMON="--arm smat --d 2 --assign content --read-mode point \
  --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 --layers 4 --d-model 256 --heads 4 --r 64 \
  --steps 5000 --nb 32 --lr 1e-3 --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_T512_frozen_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mqar_lm.py $COMMON "${@:2}" || echo "FAILED: $1"; }

run "A frozen embed + shift1 s0" --hash-src embed --hash-freeze --hash-shift 1 --balance-coef 0 --seed 0
run "B frozen embed, no shift s0" --hash-src embed --hash-freeze --hash-shift 0 --balance-coef 0 --seed 0
run "C learned hidden + shift1 s0" --hash-src hidden --hash-shift 1 --balance-coef 0.01 --seed 0
run "A frozen embed + shift1 s1" --hash-src embed --hash-freeze --hash-shift 1 --balance-coef 0 --seed 1
echo "SWEEP DONE: $(date)"

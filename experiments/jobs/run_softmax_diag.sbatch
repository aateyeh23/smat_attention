#!/bin/bash
#SBATCH --job-name=smat-sm-diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out

# The softmax control plateaued at 0.060 in the sweep; results/mqar_T512.csv has
# it at 1.000.  Isolate which of the differences is responsible:
#   A  exactly the flags of run_mqar.sbatch that produced the 1.000 row
#   B  those flags + the Mamba-2 gate flags the sweep adds
#   C  A but with the sweep's 5000-step schedule
set -uo pipefail
source experiments/prelude.sh
BASE="--arm softmax --qk-norm --short-conv 3 --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --eval-every 500 --eval-n 256 \
  --out results/mqar_softmax_diag.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=4 python -u tasks/mqar.py $BASE "${@:2}" || echo "FAILED: $1"; }
run "A original flags, 2500 steps"      --steps 2500 --tag _A_orig
run "B + mamba2 gate flags, 2500 steps" --steps 2500 --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --tag _B_gate
run "C original flags, 5000 steps"      --steps 5000 --tag _C_5k
echo "DIAG DONE: $(date)"

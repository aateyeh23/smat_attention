#!/bin/bash
#SBATCH --job-name=smat-cnt2sm0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out

# Does the rebuilt counting task admit a solution at all?  The first pass of the
# grid ran with --short-conv 1, a depthwise kernel of size 1, which mixes no
# neighbours: with one layer the answer row saw only its QMARK while the k
# requested symbols sit at t-k..t-1, so smat d=1 and softmax both sat at the
# marginal-count floor with byte-identical curves.  Width 4 reaches t-3.
#
# Three short arms, 4000 steps each.  What this has to show before the full grid
# is worth five GPU-hours: softmax off the floor (0.338 / 0.229 at k=1,3), and
# smat d=1 no longer tracking it step for step.
set -uo pipefail
source experiments/prelude.sh

C="--T 1024 --chunk 256 --n-marks 12 --n-syms 512 --pool 8 --n-ans 8 --layers 1 \
  --d-model 256 --heads 4 --r 64 --nb 32 --lr 3e-4 --steps 4000 --eval-every 1000 \
  --short-conv 4 --seed 0 --out results/count2_smoke.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/counting.py $C "${@:2}" || echo "FAILED: $1"; }

run "softmax k1" --arm softmax --d 4 --k 1
run "softmax k3" --arm softmax --d 4 --k 3
run "smat d1 k1" --arm smat --d 1 --k 1
echo "SMOKE DONE: $(date)"

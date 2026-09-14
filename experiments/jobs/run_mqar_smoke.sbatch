#!/bin/bash
#SBATCH --job-name=smat-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

# Does the stack run on the cluster at all?  200 steps of each arm we intend to sweep.
set -uo pipefail
source experiments/prelude.sh
PYTHONPATH=.:../smat python -u test_content_addr.py || { echo "EXACTNESS FAILED"; exit 1; }
COMMON="--decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 --layers 4 --d-model 256 --heads 4 --r 64 \
  --steps 200 --nb 32 --lr 1e-3 --eval-every 200 --eval-n 128 --out results/mqar_smoke.csv"
run() { echo "==== $1  $(date) ===="; OMP_NUM_THREADS=4 python -u tasks/mqar.py $COMMON "${@:2}" || echo "FAILED: $1"; }
run "softmax"            --arm softmax
run "smat d1"            --arm smat --d 1
run "smat d2 positional" --arm smat --d 2
run "smat d2 content"    --arm smat --d 2 --assign content --read-mode point
run "smat d3 content"    --arm smat --d 3 --assign content --read-mode point
run "smat d3 plane"      --arm smat --d 3 --assign content --read-mode plane
echo "SMOKE DONE: $(date)"

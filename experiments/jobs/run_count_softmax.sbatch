#!/bin/bash
#SBATCH --job-name=smat-countsm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Softmax reference for multi-group counting: no mask, every countable token
# visible, the kernel alone has to restrict the sum to the k requested symbols.
# Given a width-3 causal conv, since our arms get their grouping from the hash
# and a baseline needs the equivalent information.
set -uo pipefail
source experiments/prelude.sh
C="--T 1024 --chunk 256 --n-marks 12 --n-syms 8 --n-ans 8 --layers 1 --d-model 256 \
  --heads 4 --r 64 --nb 8 --lr 1e-3 --steps 4000 --eval-every 1000 --seed 0 --short-conv 3 \
  --out results/counting.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/counting.py $C "${@:2}" || echo "FAILED: $1"; }
for K in 1 2 3; do run "softmax k$K" --arm softmax --d 2 --k $K --tag _sc3; done
echo "COUNT SOFTMAX DONE: $(date)"

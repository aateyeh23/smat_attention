#!/bin/bash
#SBATCH --job-name=smat-mqar-c3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# Rerun of the content-addressed arms with the hash tied across the query and
# key roles (one projection on the token's layer input), which is the hypothesis
# Prop. "Content-addressed recall" actually needs.  The first version hashed the
# separate W_Q and W_K projections, so a repeated key never landed in the cell
# its own query asked for.  Also reports the measured hit rate of that
# proposition (hash_hit), which says whether the hash aligned at all.
#
# Carried over unchanged from the seed-0 sweep (results/mqar_T512_content.csv):
#   softmax 0.060 (control is broken, under separate investigation)
#   d=1 0.905   d=2 positional 0.012   d=3 positional 0.055
#   d=2 content/point 0.048 (untied hash)
set -uo pipefail
source experiments/prelude.sh
PYTHONPATH=.:../smat python -u test_content_addr.py || { echo "EXACTNESS FAILED"; exit 1; }

COMMON="--decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 --layers 4 --d-model 256 --heads 4 --r 64 \
  --steps 5000 --nb 32 --lr 1e-3 --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_T512_content3.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mqar.py $COMMON "${@:2}" || echo "FAILED: $1"; }

for S in 0 1; do
  run "d2 content point s$S" --arm smat --d 2 --assign content --read-mode point --seed $S
  run "d3 content point s$S" --arm smat --d 3 --assign content --read-mode point --seed $S
  run "d3 content plane s$S" --arm smat --d 3 --assign content --read-mode plane --seed $S
done
# the softmax control, with the flags that produced the saved 1.000 row
run "softmax A orig 2500" --arm softmax --steps 2500 --tag _Aorig
echo "SWEEP DONE: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-learn1024
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Does the learned hash still collapse where collapsing does not pay?
#
# At T=512 the learned assignment converged to a mask admitting 97-99% of the
# distant block (cell_occ 0.02, i.e. ~1.06 effective cells of 17) and scored
# 0.848.  That was not a pathology: the all-ones long-range block IS d=1, which
# scores 0.905 at T=512, so the optimiser was climbing toward a strong nearby
# solution and collapse was the cheap route there.
#
# At T=1024 that solution is worth 0.021.  If collapse is driven by the reward
# for reaching it, it should not happen here -- and every fix for it (collision
# penalty, frozen gamma, batchnormed logits, frozen->unfrozen curriculum) would
# be solving a problem that does not arise at the length that matters.  The
# content+learned configuration has never been run at T=1024.
#
# Watch cell_occ and G density, not accuracy: high accuracy with density ~1 is
# the baseline recovered through our machinery, not routing.
set -uo pipefail
source experiments/prelude.sh

BASE="--arm smat --d 2 --assign content --read-mode point --hash-shift 1 \
  --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --steps 4000 \
  --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_T1024_learned.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mqar.py $BASE "${@:2}" || echo "FAILED: $1"; }

run "learned annealed s0" --balance-coef 0.01 --anneal-steps 2500 --seed 0
run "learned annealed s1" --balance-coef 0.01 --anneal-steps 2500 --seed 1
run "learned hard STE s0" --balance-coef 0.01 --seed 0 --tag _hardste
echo "LEARN1024 DONE: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-align
#SBATCH --partition=jsteinhardt,songmei,berkeleynlp,yugroup
#SBATCH --exclude=morgoth,merry,roo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:15:00
#SBATCH --output=logs/mqar-align-%j.out

# PRIORITY 2: can the assignment be LEARNED, now that we know how it fails?
#
# At T=1024 the learned hash does not collapse -- occupancy 0.99, row density
# 0.045, which is exactly 1/N0 and perfectly selective.  It fails purely on
# ALIGNMENT: hit rate 0.045, i.e. chance.  So the anti-collapse machinery
# (collision penalty, frozen gamma, batchnormed logits) is aimed at a problem
# that does not occur here, and what is needed is pressure toward agreement.
#
#   curriculum   train with the hash frozen for 1500 steps -- the configuration
#                that reaches 0.94 -- then release it.  Starts inside the good
#                basin instead of searching for it.
#   agreement    the routing objective, which failed at T=512 for a reason that
#                does not apply here: collapse was rewarded there (an all-ones
#                mask IS d=1, worth 0.905 at T=512 and 0.02 at T=1024).
#   both         curriculum then agreement
#
# Frozen reference on the same setting: 0.976 / 0.940 / 0.934.
# Learned, unaided: 0.009.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

BASE="--arm smat --d 2 --assign content --read-mode point --hash-shift 1 \
  --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --steps 3000 \
  --eval-every 500 --eval-n 256 --early-stop 0.995 --balance-coef 0.01 \
  --out results/mqar_T1024_align_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mqar_lm.py $BASE "${@:2}" || echo "FAILED: $1"; }

run "curriculum unfreeze@1500" --hash-src embed --hash-freeze --unfreeze-at 1500 --seed 0
run "agreement objective"      --anneal-steps 1500 --agree-coef 1.0 --seed 0
run "curriculum + agreement"   --hash-src embed --hash-freeze --unfreeze-at 1500 --agree-coef 1.0 --seed 0
echo "ALIGN DONE: $(date)"

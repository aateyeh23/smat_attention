#!/bin/bash
#SBATCH --job-name=smat-lhash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Can the LEARNED CONTEXTUAL hash be trained?  This is now the critical path.
#
# Softmax solves every synthetic here given its own budget, so the contribution
# is cost rather than capability -- which means the application is long-context
# language, where a vocabulary table is useless (matching is semantic, not
# exact-symbol) and only a contextual hash can work.  That configuration is the
# one that fails: 0.009 from scratch here, and in the collaborator's Zoology runs it
# reaches 86.1 at d=2 but collapses mid-run at d=3 (72 -> 32) and d=4 (42 -> 10).
#
# Two ingredients exist separately and have never been combined:
#   * his learned key-side conv, which replaces the hand-set offset and is what
#     makes the hash engage at all (86.1 with conv against 37.8 with shift 1);
#   * the agreement objective from this branch, which rewards a query for placing
#     mass where its payload sits.  It is absent from v10, and it is what stopped
#     the drift here: curriculum alone degraded to 0.856, curriculum + agreement
#     reached 0.973 against the frozen assignment's 0.950.
#
# The collapses look like the same failure mode we diagnosed -- a hash with
# nothing pulling it toward agreement drifts to whatever the balance term wants.
# So: conv + anneal + balance (v10-like), crossed with agreement and curriculum.
#
# References on this setting, MQAR T=1024 d=2: id hash 0.999 / 1.000 / 0.999,
# frozen embedding projection 0.976 / 0.940 / 0.934, learned from scratch 0.009.
set -uo pipefail
source experiments/prelude.sh

BASE="--arm smat --d 2 --assign content --read-mode point --hash-src hidden --hash-conv 4 \
  --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --steps 4000 \
  --eval-every 500 --eval-n 256 --early-stop 0.995 --balance-coef 0.01 \
  --out results/learnedhash.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mqar.py $BASE "${@:2}" || echo "FAILED: $1"; }

run "v10-like: conv + anneal"        --anneal-steps 1500 --seed 0 --tag _v10
run "+ agreement"                   --anneal-steps 1500 --agree-coef 1.0 --seed 0 --tag _v10ag
run "+ curriculum"                  --anneal-steps 1500 --unfreeze-at 1500 --seed 0 --tag _v10cur
run "+ agreement + curriculum"      --anneal-steps 1500 --agree-coef 1.0 --unfreeze-at 1500 --seed 0 --tag _v10both
run "fixed shift control (no conv)" --hash-conv 0 --hash-shift 1 --anneal-steps 1500 --agree-coef 1.0 --seed 0 --tag _sh1ag
echo "LHASH DONE: $(date)"

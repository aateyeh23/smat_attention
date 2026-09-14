#!/bin/bash
#SBATCH --job-name=smat-zoofair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# The Zoology-generator comparison at a budget every arm can use.
#
# The first pass gave ours 0.982 against softmax 0.454 at T=256/p16, and 0.395
# against 0.000 at T=512/p64 -- but softmax ran at lr 1e-3 for 2500 steps, and
# softmax on these tasks has a phase transition that lr 1e-3 never reaches (it
# sat at chance for 20k steps on multi-item recall, then solved the task in 8k
# at lr 3e-4).  Softmax does converge at lr 1e-3 on our own MQAR layout, so the
# earlier 0.998 ceiling stands; Arora et al.'s generator is harder -- power-law
# query gaps, every key queried exactly once -- and there is no reason to think
# the same budget carries over.
#
# So: identical budget for all three arms, at the learning rate known to let
# softmax transition.  A win over an undertrained baseline is not a win.
set -uo pipefail
source experiments/prelude.sh

COMMON="--zoology --qk-norm --short-conv 3 --layers 4 --d-model 256 --heads 4 --r 64 \
  --nb 32 --lr 3e-4 --steps 8000 --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --seed 0 --out results/zoo_fair.csv"
GATE="--decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0"
CONTENT="--assign content --read-mode point --hash-src id --hash-shift 1 --balance-coef 0"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mqar.py $COMMON "${@:2}" || echo "FAILED: $1"; }

for CFG in "256 16" "512 64"; do
  set -- $CFG; TT=$1; NP=$2
  run "T$TT p$NP softmax" --arm softmax --T $TT --chunk $((TT/4)) --n-pairs $NP --tag _fair
  run "T$TT p$NP d1"      --arm smat --d 1 $GATE --T $TT --chunk $((TT/4)) --n-pairs $NP --tag _fair
  run "T$TT p$NP content" --arm smat --d 2 $GATE $CONTENT --T $TT --chunk $((TT/4)) --n-pairs $NP --tag _fair
done
echo "ZOOFAIR DONE: $(date)"

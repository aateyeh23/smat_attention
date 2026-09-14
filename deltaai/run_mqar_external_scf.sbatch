#!/bin/bash
#SBATCH --job-name=smat-extern
#SBATCH --partition=jsteinhardt,songmei,berkeleynlp,yugroup
#SBATCH --exclude=morgoth,merry,roo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/mqar-extern-%j.out

# Making the MQAR result an external comparison rather than an internal one.
#
# Two gaps.  (1) Every T=1024 run used this repo's own MQAR class, not the
# Zoology generator (Arora et al.), whose layout differs -- power-law query gaps,
# every key queried exactly once, vocabulary split into key and value halves.
# The --zoology flag reproduces it and we never used it.  (2) There is no
# softmax ceiling at T=1024, so "linear attention fails at 0.021" is unanchored:
# we do not know whether full attention solves the task at that length either.
#
# Softmax gets 2500 steps, the budget at which it converges; a 5000-step
# schedule left it at 0.060 in earlier runs and that is a property of the
# schedule, not the model.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

COMMON="--qk-norm --short-conv 3 --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 \
  --lr 1e-3 --eval-every 500 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_external_scf.csv"
GATE="--decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0"
CONTENT="--assign content --read-mode point --hash-src id --hash-shift 1 --balance-coef 0"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mqar_lm.py $COMMON "${@:2}" || echo "FAILED: $1"; }

# (2) the missing ceiling, our layout, T=1024
run "softmax ceiling T1024" --arm softmax --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 \
    --n-keys 128 --steps 2500 --seed 0 --tag _ceil

# (1) the Zoology generator, two of its standard shapes
for CFG in "256 16" "512 64"; do
  set -- $CFG; TT=$1; NP=$2
  run "zoology T$TT p$NP softmax" --zoology --arm softmax --T $TT --chunk $((TT/4)) \
      --n-pairs $NP --steps 2500 --seed 0 --tag _zoo
  run "zoology T$TT p$NP d1"      --zoology --arm smat --d 1 $GATE --T $TT --chunk $((TT/4)) \
      --n-pairs $NP --steps 5000 --seed 0 --tag _zoo
  run "zoology T$TT p$NP content" --zoology --arm smat --d 2 $GATE $CONTENT --T $TT \
      --chunk $((TT/4)) --n-pairs $NP --steps 5000 --seed 0 --tag _zoo
done
echo "EXTERN DONE: $(date)"

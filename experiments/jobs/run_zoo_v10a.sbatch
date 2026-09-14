#!/bin/bash
#SBATCH --job-name=smat-zooA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# A control, not an attempt to make the learned hash work.
#
# a collaborator reports the v10 scheme reaching 86.1 at d=2 on ZOOLOGY MQAR, with the
# learned key-side conv as the ingredient that makes it engage at all (86.1 with
# conv against 37.8 with shift 1).  Last night's job ran the same scheme on this
# branch's plain MQAR -- 16 pairs, 128 keys, T=1024, no --zoology -- and every
# arm sat at chance, 0.010 to 0.019.  The two of us have therefore never compared
# the same setting, and this branch's own multi-key result has the OPPOSITE sign
# on the conv: replacing the fixed offset cost rather than helped, 0.000 at both
# d=3,k=2 and d=4,k=3.
#
# So: run the scheme under Zoology's generator on this code, with a frozen id
# hash alongside as the control that the setting is solvable here at all.  What
# this cannot settle is the collaborator's exact pair count and vocabulary, which we do not
# have; if they differ from the defaults below the comparison is still not
# controlled and we should ask him for the config rather than guess again.
set -uo pipefail
source experiments/prelude.sh

BASE="--arm smat --d 2 --zoology --assign content --read-mode point \
  --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 1024 --chunk 256 --n-pairs 16 --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 \
  --lr 1e-3 --steps 4000 --eval-every 500 --eval-n 256 --early-stop 0.995 \
  --out results/zoo_v10.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mqar.py $BASE "${@:2}" || echo "FAILED: $1"; }

run "zoo frozen id hash (control)" --hash-src id --hash-shift 1 --seed 0 --tag _zooid
run "zoo v10-like: conv + anneal"  --hash-src hidden --hash-conv 4 --anneal-steps 1500 \
    --balance-coef 0.01 --seed 0 --tag _zoov10
echo "ZOO A DONE: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-mqar-table
#SBATCH --partition=yugroup,jsteinhardt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/mqar-table-%j.out

# The MQAR table, every arm at one commit, two seeds.  Earlier numbers for the
# positional and d=1 arms came from the first code version of the night; nothing
# changed tonight touches them (every edit was inside ContentAssign or gated on
# it existing), but the table should not rest on that argument.
#
# New this run: g_density, the share of the distant block a recent query can
# actually see.  1/N0 = 0.059 at d=2 is a selective mask; 1.0 is no mask at all.
# cell_occ cannot tell a collapsed hash from one that has swept the noise into
# its own cell; this can, and it is what decides whether the d=2 content result
# is routing or dressed-up linear attention.
#
# softmax is rerun with the flags that produced the saved 1.000 row (2500 steps,
# no gate flags) since the sweep's 5000-step schedule left it at 0.060.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

BASE="--qk-norm --short-conv 3 --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --eval-every 1000 --eval-n 256 \
  --early-stop 0.995 --out results/mqar_T512_table_scf.csv"
GATE="--decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mqar_lm.py $BASE "${@:2}" || echo "FAILED: $1"; }

for S in 0 1; do
  run "softmax s$S"              --arm softmax --steps 2500 --seed $S
  run "d1 linear attn s$S"       --arm smat --d 1 $GATE --steps 5000 --seed $S
  run "d2 positional s$S"        --arm smat --d 2 $GATE --steps 5000 --seed $S
  run "d3 positional s$S"        --arm smat --d 3 $GATE --steps 5000 --seed $S
  run "d2 content anneal s$S"    --arm smat --d 2 $GATE --steps 5000 --assign content --read-mode point \
                                 --hash-shift 1 --balance-coef 0.01 --anneal-steps 2500 --seed $S
  run "d3 content anneal s$S"    --arm smat --d 3 $GATE --steps 5000 --assign content --read-mode point \
                                 --hash-shift 1 --balance-coef 0.01 --anneal-steps 2500 --seed $S
  run "d3 content plane s$S"     --arm smat --d 3 $GATE --steps 5000 --assign content --read-mode plane \
                                 --hash-shift 1 --balance-coef 0.01 --anneal-steps 2500 --seed $S
  run "d2 content frozen s$S"    --arm smat --d 2 $GATE --steps 5000 --assign content --read-mode point \
                                 --hash-src embed --hash-freeze --hash-shift 1 --balance-coef 0 --seed $S
done
echo "TABLE DONE: $(date)"

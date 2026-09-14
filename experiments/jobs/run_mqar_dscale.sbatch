#!/bin/bash
#SBATCH --job-name=smat-dscale
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out

# Does d buy anything on MQAR once the hash is correct?  Everything so far used
# the frozen hash only at d=2, where dim=1 and a hyperplane of F_q^1 IS a point,
# so point and plane modes are the same object and nothing could distinguish them.
#
# T=1024 is the smallest length at which d=2,3,4 are all admissible: at T=512,
# d=4 has 2B=310 > T_R=256 and the VC lower bound does not certify.
#   d=2: q=23  N0=23   B=23   1 point per plane   (point == plane)
#   d=3: q=11  N0=121  B=132  11 points per plane
#   d=4: q=5   N0=125  B=155  25 points per plane
#
# Predictions to check against g_density: point mode admits 1/N0 (0.043, 0.008,
# 0.008), plane mode admits 1/q (0.091, 0.200).  Selectivity should improve with
# d for point types and degrade for plane types.
#
# Frozen hash throughout: the assignment is removed from the optimisation so this
# measures the mask, not the learning problem.  Seed 0 runs to completion first,
# so killing the job still leaves a complete seed.
set -uo pipefail
source experiments/prelude.sh

BASE="--decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --steps 4000 \
  --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_T1024_dscale.csv"
FRZ="--assign content --read-mode point --hash-src embed --hash-freeze --hash-shift 1 --balance-coef 0"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mqar.py $BASE "${@:2}" || echo "FAILED: $1"; }

for S in 0 1; do
  run "d1 reference s$S"        --arm smat --d 1 --seed $S
  run "d3 positional s$S"       --arm smat --d 3 --seed $S
  run "d2 frozen point s$S"     --arm smat --d 2 $FRZ --seed $S
  run "d3 frozen point s$S"     --arm smat --d 3 $FRZ --seed $S
  run "d3 frozen plane s$S"     --arm smat --d 3 $FRZ --read-mode plane --seed $S
  run "d4 frozen point s$S"     --arm smat --d 4 $FRZ --seed $S
  run "d4 frozen plane s$S"     --arm smat --d 4 $FRZ --read-mode plane --seed $S
done
echo "DSCALE DONE: $(date)"

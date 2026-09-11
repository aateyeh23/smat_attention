#!/bin/bash
#SBATCH --job-name=smat-posgrid
#SBATCH --partition=yugroup,jsteinhardt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/mqar-posgrid-%j.out

# The missing cell of the (assignment x type family) grid: positional profiles
# read through POINT types.  Every positional run so far used hyperplane types
# (the published rho(i) = i mod B); every point-type run so far was
# content-addressed.  Without this cell one cannot tell whether the content
# addressing or the narrower type is what matters.
#
# Prediction: at chance, and worse than positional/hyperplane.  A positional
# point type admits one profile of N0, so the payload is visible with probability
# 1/N0 (0.008 at d=3) rather than 1/q (0.09).  Narrowing the type without making
# the assignment content-dependent should make things strictly worse, which is
# the point of running it.
#
# Same setting as job 3530122 (T=1024, chunk 256) so the rows are comparable.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

BASE="--arm smat --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --steps 4000 \
  --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_T1024_posgrid_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mqar_lm.py $BASE "${@:2}" || echo "FAILED: $1"; }

for S in 0 1; do
  run "d3 positional point s$S" --d 3 --pos-read point --seed $S
  run "d4 positional point s$S" --d 4 --pos-read point --seed $S
  run "d4 positional plane s$S" --d 4 --pos-read plane --seed $S
done
echo "POSGRID DONE: $(date)"

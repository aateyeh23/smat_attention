#!/bin/bash
#SBATCH --job-name=smat-mkid2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Third seed of the staircase table, which currently carries one.  The
# limitations paragraph already concedes two seed artefacts in this project -- a
# d=3 arm reading 0.450 against 0.835 and 0.855, and a d=2 over d=3 ordering that
# vanishes into noise once averaged -- so the central table is the one place a
# single seed is least affordable.
#
# Config is copied verbatim from run_smat-mkid.sbatch, seed apart: changing the
# budget mid-table would confound the comparison with seed 0.  That run took 34
# minutes for the nine cells.
set -uo pipefail
source experiments/prelude.sh

MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 8 --lr 1e-3 --steps 4000 --eval-every 1000 --dirs oracle \
  --hash-src id --read-mode plane --out results/mkar_id.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py $MK "${@:2}" || echo "FAILED: $1"; }

for D in 2 3 4; do
  for K in 1 2 3; do
    run "d$D k$K s2" --d $D --k $K --seed 2
  done
done
echo "MKID SEED2 DONE: $(date)"

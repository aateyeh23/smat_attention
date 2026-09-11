#!/bin/bash
#SBATCH --job-name=smat-memmatch
#SBATCH --partition=jsteinhardt,yugroup
#SBATCH --exclude=balrog,luthien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=14:00:00
#SBATCH --output=logs/mqar-memmatch-%j.out

# The control the headline claim needs.  "Content addressing 0.976 vs linear
# attention 0.021 at T=1024" compares a mask holding N0 = 23 states per head
# against a recurrence holding ONE, so it conflates content addressing with 23x
# the memory.  Scaling the feature rank to r = 23 * 64 = 1472 gives linear
# attention a state of exactly 1472 * 65 = 95,680 words per head -- the same as
# 23 states of rank 64 -- and asks whether memory alone accounts for the gap.
#
# If this solves MQAR, the claim reduces to "more memory helps" and the paper
# has to be re-framed around access cost instead: the bucketed read is O(rp) per
# token where this baseline is O(N0 rp), a 23x difference at equal memory.
# If it does not, content addressing is doing something memory cannot buy.
#
#   r=256   gated    intermediate point, 4x memory
#   r=1472  gated    memory-matched to d=2 content (gated comparison)
#   r=1472  ungated  memory-matched, and matched to the 0.995 ungated result
#
# Same setting as job 3530122 so rows are comparable: T=1024, chunk 256, n=512.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

BASE="--arm smat --d 1 --qk-norm --short-conv 3 \
  --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --nb 32 --lr 1e-3 --steps 4000 \
  --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_T1024_memmatch_scf.csv"
GATE="--decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mqar_lm.py $BASE "${@:2}" || echo "FAILED: $1"; }

for S in 0 1; do
  run "r1472 gated s$S"    --r 1472 $GATE --seed $S --tag _r1472
  run "r1472 ungated s$S"  --r 1472        --seed $S --tag _r1472_nogate
  run "r256 gated s$S"     --r 256  $GATE  --seed $S --tag _r256
done
echo "MEMMATCH DONE: $(date)"

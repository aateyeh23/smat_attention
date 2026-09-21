#!/bin/bash
#SBATCH --job-name=smat-basegap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/basegap-%j.out
# Completes two baseline comparisons already in results/ but missing cells:
#   lsh      the point read (one bucket per query = LSH attention) at d=4, k=2,3
#            under the fair2 multi-key protocol, to finish the LSH column
#   memmatch gated linear attention with the SMAT d=2 memory in ONE state
#            (r=1472), seed 1, and the 4x-memory midpoint r=256
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
[ -f experiments/site.conf ] && . experiments/site.conf
[ -n "${SMAT_ENV:-}" ] && eval "$SMAT_ENV"
echo "start $(date)"
MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 32 --lr 3e-4 --steps 12000 --eval-every 2000 --hash-src id --short-conv 3 \
  --seed 0 --read-mode point --dirs oracle --tag _fair2 --out results/mkar_fair2_point_gap.csv"
MQ="--arm smat --d 1 --qk-norm --short-conv 3 --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 \
  --n-keys 128 --layers 4 --d-model 256 --heads 4 --nb 32 --lr 1e-3 --steps 4000 --eval-every 1000 \
  --eval-n 256 --early-stop 0.995 --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 \
  --out results/mqar_T1024_memmatch_gap.csv"
case "$1" in
  lsh)      python -u tasks/mkar.py $MK --d 4 --k 2 & python -u tasks/mkar.py $MK --d 4 --k 3 & ;;
  memmatch) python -u tasks/mqar.py $MQ --r 1472 --seed 1 --tag _r1472 & python -u tasks/mqar.py $MQ --r 256 --seed 0 --tag _r256 & ;;
  *) echo "usage: $0 lsh|memmatch"; exit 2 ;;
esac
wait
echo "done $(date)"

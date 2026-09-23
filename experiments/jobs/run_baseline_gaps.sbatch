#!/bin/bash
#SBATCH --job-name=smat-basegap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/basegap-%j.out
# Completes the LSH column of the multi-key table: the point read (one bucket
# per query = LSH attention) at d=4, k=2,3 under the fair2 protocol.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
[ -f experiments/site.conf ] && . experiments/site.conf
[ -n "${SMAT_ENV:-}" ] && eval "$SMAT_ENV"
echo "start $(date)"
MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 32 --lr 3e-4 --steps 12000 --eval-every 2000 --hash-src id --short-conv 3 \
  --seed 0 --read-mode point --dirs oracle --tag _fair2 --out results/mkar_fair2_point_gap.csv"
case "$1" in
  lsh)      python -u tasks/mkar.py $MK --d 4 --k 2 & python -u tasks/mkar.py $MK --d 4 --k 3 & ;;
  *) echo "usage: $0 lsh"; exit 2 ;;
esac
wait
echo "done $(date)"

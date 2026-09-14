#!/bin/bash
#SBATCH --job-name=smat-replicate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Seeds for the three comparisons the paper will actually quote.  Job 3530122
# established them on seed 0 and was cancelled before seed 1: at T=1024 frozen,
# d=2 point 0.976, d=3 point 0.935, d=3 plane 0.450, d=1 reference 0.021.  The
# d-ordering rests on 4-point gaps from a single seed, which is exactly the size
# a seed can invent -- and the baselines in this setting have already shown
# bimodal behaviour (softmax 1.000 / 0.034 at T=512), so d=1 needs replication
# as much as the treatment arms do.
#
# d=4 is dropped: 0.976 > 0.935 > 0.865 already establishes the monotone decline,
# and N0=125 makes it the most expensive arm under the dense implementation.
set -uo pipefail
source experiments/prelude.sh

BASE="--arm smat --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --steps 4000 \
  --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_T1024_replicate.csv"
FRZ="--assign content --read-mode point --hash-src embed --hash-freeze --hash-shift 1 --balance-coef 0"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mqar.py $BASE "${@:2}" || echo "FAILED: $1"; }

for S in 1 2; do
  run "d1 reference s$S"    --d 1 --seed $S
  run "d2 frozen point s$S" --d 2 $FRZ --seed $S
  run "d3 frozen point s$S" --d 3 $FRZ --seed $S
  run "d3 frozen plane s$S" --d 3 $FRZ --read-mode plane --seed $S
done
echo "REPLICATE DONE: $(date)"

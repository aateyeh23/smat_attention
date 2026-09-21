#!/bin/bash
#SBATCH --job-name=smat-mkar-rec
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# The VC-1 recurrent baselines on multi-key subset recall, matching the set on
# d-Subset Routing: Mamba-2 (SSD), DeltaNet, Gated DeltaNet, log-linear attention.
#
# Same block as every other arm -- short conv 3, QKV/out projections, QK-norm,
# 4 heads, MLP -- with the mixer swapped for the recurrence (RecurrentMixer in
# src/smat/model.py, running the chunked forms of src/smat/mixers/recurrent.py
# that the routing sweep uses).  Protocol is fair2 verbatim, the one behind every
# SMAT cell in the table; --d 2 fixes the task geometry exactly as the softmax
# baseline does.  All three seeds at k = 1, 2, 3.
#
#   sbatch run_mkar_recurrent.sbatch mamba2
#   sbatch run_mkar_recurrent.sbatch deltanet
#   sbatch run_mkar_recurrent.sbatch gated_deltanet
#   sbatch run_mkar_recurrent.sbatch loglinear
set -uo pipefail
source experiments/prelude.sh

ARM="${1:?arm}"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py "${@:2}" || echo "FAILED: $1"; }
for SEED in 0 1 2; do for K in 1 2 3; do
  run "$ARM s$SEED k$K" --T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 \
      --d-model 256 --heads 4 --r 64 --nb 32 --lr 3e-4 --steps 12000 \
      --eval-every 2000 --hash-src id --short-conv 3 --seed $SEED --tag _fair2 \
      --out results/mkar_fair2_rec_${ARM}.csv --arm $ARM --d 2 --k $K
done; done
echo "MKAR RECURRENT $ARM DONE: $(date)"

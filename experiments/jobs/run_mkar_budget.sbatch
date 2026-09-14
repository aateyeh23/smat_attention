#!/bin/bash
#SBATCH --job-name=smat-mkbudget
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Can the baselines learn this task AT ALL, given a fair budget?
#
# At batch 8 and 4000 steps both softmax and linear attention sat at loss
# 0.3756-0.3789 for the whole run -- exactly the BCE of predicting all zeros for
# one positive in eight.  Not slow learning: no learning.  Our arms reach loss
# 0.001 by step 1000 because the mask removes the search; a baseline has to FIND
# the induction match, and induction heads are slow to form.  ~32k sequences may
# simply be too few to escape the all-negative predictor.
#
# So this tests the easiest cell (k=1, single-item retrieval, which full
# attention certainly can express) with 5x the batch, 5x the steps and two
# learning rates.  Reporting "the baselines fail" without this would be claiming
# a capability result from a budget artefact -- the same error as withholding
# the short conv, in a different disguise.
#
# If softmax still cannot do k=1 at 20k steps, the task's readout or layer count
# is the binding constraint and the whole comparison needs rethinking, not the
# baselines.
set -uo pipefail
source experiments/prelude.sh

MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 32 --steps 20000 --eval-every 4000 --hash-src id --short-conv 3 \
  --k 1 --seed 0 --out results/mkar_budget.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py $MK "${@:2}" || echo "FAILED: $1"; }

run "softmax lr1e-3" --arm softmax --d 2 --lr 1e-3 --tag _lr1e-3
run "softmax lr3e-4" --arm softmax --d 2 --lr 3e-4 --tag _lr3e-4
run "d1 linear lr1e-3" --arm smat --d 1 --lr 1e-3 --tag _lr1e-3
run "ours d3 lr1e-3 (reference, same budget)" --arm smat --d 3 --lr 1e-3 --tag _ref
echo "BUDGET DONE: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-routing-base2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# The two baselines the paper names as VC 1 and did not measure on d-Subset
# Routing: Gated DeltaNet and log-linear attention.
#
# Both are claimed by Prop. vc1 to sit at d = 1, and the claim rests on the row
# support alone: Gated DeltaNet carries one gated state and log-linear
# attention carries Theta(log T) of them, and neither can drop an element of
# the prefix -- the decay is an exponential and the block weights are softplus,
# so both are strictly positive.  Running them turns two predictions into two
# measurements.  Implementations are in src/smat/mixers/recurrent.py, whose
# self-test checks each chunked form against its sequential rule and checks that
# the hierarchical mask covers exactly the causal triangle; it runs first here.
#
# Flags match run_routing.sbatch and run_routing_recurrent.sbatch exactly --
# same T, d list, kmax, steps, batch, lr, seeds and seed -- so choose_marked_set
# draws the same marked columns and these rows concatenate with
# results/routing_learned.csv.  The arms are mask-free, so like the softmax and
# recurrent arms they run once, on d = 1's marked set.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source experiments/prelude.sh

python -u src/smat/mixers/recurrent.py

python -u tasks/routing_train.py \
    --T 1024 --d 1 2 3 4 5 --kmax 6 \
    --arms gated_deltanet loglinear \
    --steps 1500 --nb 64 --lr 3e-3 --seeds 3 \
    --device cuda \
    --out results/routing_learned_gdn_loglinear.csv

echo "done: $(date)"

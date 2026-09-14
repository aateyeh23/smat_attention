#!/bin/bash
#SBATCH --job-name=smat-routing-rec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# The two recurrent baselines on d-Subset Routing: Mamba-2's scalar-decay
# recurrence and the delta rule, in the same one-layer Router as the other arms
# (src/smat/mixers/recurrent.py, chunked forms checked against the sequential rule).
#
# Flags match run_routing.sbatch exactly -- same T, d list, kmax, steps, batch,
# lr, seeds and seed -- so choose_marked_set draws the same marked columns and
# these rows can be concatenated with results/routing_learned.csv.  The arms are
# mask-free, so like the softmax arms they run once, on d = 1's marked set.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source experiments/prelude.sh

python -u src/smat/mixers/recurrent.py

python -u tasks/routing_train.py \
    --T 1024 --d 1 2 3 4 5 --kmax 6 \
    --arms mamba2 deltanet \
    --steps 1500 --nb 64 --lr 3e-3 --seeds 3 \
    --device cuda \
    --out results/routing_learned_recurrent.csv

echo "done: $(date)"

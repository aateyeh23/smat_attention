#!/bin/bash
#SBATCH --job-name=smat-mqar-learn
#SBATCH --partition=yugroup,jsteinhardt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mqar-learn-%j.out

# Can the assignment be LEARNED?  The frozen embedding hash + shift already solves
# MQAR (0.997 @ step 4000, results/mqar_T512_frozen_scf.csv), so the mask and the
# read schedule are fine; what is left is whether a learned hash gets there too.
#
# Two additions over the runs that failed, both aimed at the missing coupling:
#   * an annealed soft read over the 2^{dim} cells adjacent to the query's logits,
#     so the gradient is not confined to bin boundaries.  Scoring happens in the
#     hash space and only 2^{dim} states of size rp are contracted, so this is
#     O(1) in T -- it is not the O(N0) soft read, and eq. (3.x) still holds.
#     anneal reaches 1.0 (the exact hard binary mask) at step 2500, so the evals
#     at 3000/4000/5000 are hard-mask numbers; 1000/2000 are not.
#   * a routing objective: the query's cell distribution should carry mass on the
#     cell where its payload actually sits.  Nothing in the task loss or the
#     balance loss says this, which is why the router previously drifted to
#     whatever the balance term preferred.
#
#   L1  hard STE + balance            (control: does the shift alone rescue learning?)
#   L2  + annealed soft read
#   L3  + routing objective           (seeds 0 and 1)
#
# d = 2 point mode throughout.  Baselines already measured are not repeated.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

COMMON="--arm smat --d 2 --assign content --read-mode point --hash-src hidden --hash-shift 1 \
  --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 --layers 4 --d-model 256 --heads 4 --r 64 \
  --steps 5000 --nb 32 --lr 1e-3 --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_T512_learn_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mqar_lm.py $COMMON "${@:2}" || echo "FAILED: $1"; }

run "L1 hard STE + balance s0"        --balance-coef 0.01 --seed 0
run "L2 + annealed soft read s0"      --balance-coef 0.01 --anneal-steps 2500 --seed 0
run "L3 + routing objective s0"       --balance-coef 0.01 --anneal-steps 2500 --agree-coef 1.0 --seed 0
run "L3 + routing objective s1"       --balance-coef 0.01 --anneal-steps 2500 --agree-coef 1.0 --seed 1
echo "SWEEP DONE: $(date)"

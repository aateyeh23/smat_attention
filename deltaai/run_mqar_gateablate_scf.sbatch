#!/bin/bash
#SBATCH --job-name=smat-gate-ablate
#SBATCH --partition=yugroup,jsteinhardt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/mqar-gateablate-%j.out

# Is the Mamba-2 gate on the two causal blocks doing any of the work?
#
# Every arm so far carries it, because without it the ungated linear-attention
# baseline is at chance (0.015): 85-90% of the normaliser at a query row is noise
# mass.  But the arm that solves the task keeps noise_share at ~0.89 *while*
# reaching 0.985, which suggests the mask, not the gate, is what retrieves the
# payload.  If so the gate should be removable from the content-addressed arms
# and not from the others.
#
# This job runs the UNGATED half of a 2x2; the gated half is job 3529457
# (results/mqar_T512_table_scf.csv), same commit, same seeds.
#
#   d=1 linear attention        expected to collapse (reference: 0.015 ungated)
#   d=2 positional              expected to stay at chance either way
#   d=2 content, annealed       the question: 0.985 gated -> ?
#   d=2 content, frozen hash    the question: 0.997 gated -> ?
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

BASE="--qk-norm --short-conv 3 --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --steps 5000 \
  --eval-every 1000 --eval-n 256 --early-stop 0.995 --tag _nogate \
  --out results/mqar_T512_gateablate_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mqar_lm.py $BASE "${@:2}" || echo "FAILED: $1"; }

for S in 0 1; do
  run "d1 ungated s$S"                --arm smat --d 1 --seed $S
  run "d2 positional ungated s$S"     --arm smat --d 2 --seed $S
  run "d2 content anneal ungated s$S" --arm smat --d 2 --assign content --read-mode point \
                                      --hash-shift 1 --balance-coef 0.01 --anneal-steps 2500 --seed $S
  run "d2 content frozen ungated s$S" --arm smat --d 2 --assign content --read-mode point \
                                      --hash-src embed --hash-freeze --hash-shift 1 --balance-coef 0 --seed $S
done
echo "ABLATION DONE: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-idhash
#SBATCH --partition=jsteinhardt,songmei,berkeleynlp,yugroup
#SBATCH --exclude=morgoth,merry,roo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/mqar-idhash-%j.out

# Is the random projection doing anything?
#
# In the working configuration the hash is a FROZEN random projection of the
# TOKEN EMBEDDING.  Composed, that is an arbitrary fixed map from a vector which
# is itself a learned function of the token id -- and on these tasks the thing
# being matched IS token identity, the query token and the key token being the
# same symbol.  So the whole apparatus may be computing nothing more than "file
# this token under a pseudorandom function of its id", which --hash-src id does
# directly, with no projection and no embedding dependence.
#
# The projection's one extra capability is that learned embeddings can move a
# token between cells even with W frozen.  We measured that freedom being used
# badly: row density drifts UP during training (0.39 -> 0.56 at d=2, 0.06 ->
# 0.47 at d=3), i.e. toward a less selective mask.
#
# Predictions: accuracy equal or better than the frozen embedding hash
# (0.976 / 0.940 / 0.934 at d=2), and density FLAT across training, since
# nothing can move a token between cells.  A smoke run already shows hash hit
# 1.000 against the embedding hash's 0.922.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

BASE="--arm smat --assign content --read-mode point --hash-src id --hash-shift 1 \
  --balance-coef 0 --decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 1024 --chunk 256 --n-pairs 16 --n-queries 16 --n-keys 128 \
  --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --steps 4000 \
  --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_T1024_idhash_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mqar_lm.py $BASE "${@:2}" || echo "FAILED: $1"; }

for S in 0 1 2; do run "id hash d2 s$S" --d 2 --seed $S; done
run "id hash d3 s0" --d 3 --seed 0
echo "IDHASH DONE: $(date)"

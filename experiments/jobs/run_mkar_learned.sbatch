#!/bin/bash
#SBATCH --job-name=smat-mklrn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# A LEARNED hash on multi-item recall.  Every multi-item run so far has used a
# frozen one -- tasks/mkar.py hardcoded hash_freeze -- so the task where d actually
# binds has never been tested with a trainable assignment, which is the only
# version that could transfer to text.
#
# The hash is a learned projection of the token EMBEDDING rather than of the
# hidden state.  That is a real restriction and it is forced: the oracle computes
# hyperplane directions from cells_of(emb(requested keys)), so a contextual hash
# would leave the oracle reading a different cell map than the mask uses -- the
# same mismatch that silently flattened the staircase when the id hash was first
# wired in.  A contextual hash needs either a two-pass oracle or a learned
# direction map, and the learned direction map already fails (0.002).
#
# New here: a key-side agreement term.  The oracle derives directions from
# key-token cells while the mask files payloads by the convolved key-side input,
# and once the hash is learned nothing makes those agree.  The term pushes a
# payload's cell toward the cell of the key naming it.  On single-item recall the
# analogous term was decisive -- curriculum alone degraded to 0.856, curriculum
# plus agreement reached 0.973 against the frozen assignment's 0.950.
#
# References, id hash / frozen: d=3 k=2 0.979 / 0.955,  d=4 k=3 0.986 / 0.838.
set -uo pipefail
source experiments/prelude.sh

MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 8 --lr 1e-3 --steps 4000 --eval-every 1000 --dirs oracle --hash-src embed \
  --out results/mkar_learned.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py $MK "${@:2}" || echo "FAILED: $1"; }

for CELL in "3 2" "4 3"; do
  set -- $CELL; D=$1; K=$2
  run "d$D k$K learned+conv"           --d $D --k $K --hash-freeze 0 --hash-conv 4
  run "d$D k$K learned+conv+agree"     --d $D --k $K --hash-freeze 0 --hash-conv 4 --agree-coef 1.0
  run "d$D k$K curriculum+agree"       --d $D --k $K --hash-freeze 1 --hash-conv 4 --agree-coef 1.0 --unfreeze-at 1500
  run "d$D k$K learned, fixed shift"   --d $D --k $K --hash-freeze 0
done
echo "MKLRN DONE: $(date)"

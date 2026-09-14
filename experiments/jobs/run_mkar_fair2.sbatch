#!/bin/bash
#SBATCH --job-name=smat-mkar-fair2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# Multi-key subset recall, every arm under ONE protocol.
#
# The baselines collected so far are not usable.  mkar_baselines and
# mkar_fairbase ran at batch 8, where softmax never escapes its plateau
# (exact_support 0.000, loss pinned at 0.376, through 16k steps at lr 3e-4).
# mkar_budget ran the same configuration at batch 32 and softmax reached 1.000
# by step 8000.  The difference is the batch, not the task, so every "softmax
# fails" cell at batch 8 is an artefact -- including the ones the fairbase job
# was written to settle.
#
# This runs batch 32, lr 3e-4, 12000 steps for all three arms: full causal
# attention, the single-state linear control (d=1), and the hyperplane read at
# d = 2, 3, 4.  k = 1, 2, 3 throughout.  The question it settles is whether
# softmax, given a budget that lets it train, also answers k >= 2 -- if it does,
# the k <= d-1 staircase describes our construction rather than the difficulty
# of the task, and the caption has to say so.
#
# Three groups so they can be submitted to three GPUs:
#   sbatch run_mkar_fair2.sbatch base     # softmax and d=1, k = 1,2,3
#   sbatch run_mkar_fair2.sbatch smat23   # d = 2, 3, k = 1,2,3
#   sbatch run_mkar_fair2.sbatch smat4    # d = 4,    k = 1,2,3
set -uo pipefail
source experiments/prelude.sh

GROUP="${1:-base}"
MK="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 32 --lr 3e-4 --steps 12000 --eval-every 2000 --hash-src id --short-conv 3 \
  --seed 0 --tag _fair2 --out results/mkar_fair2_${GROUP}.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py $MK "${@:2}" || echo "FAILED: $1"; }

case "$GROUP" in
  base)
    for K in 1 2 3; do
      run "softmax k$K"   --arm softmax --d 2 --k $K
      run "d1 linear k$K" --arm smat --d 1 --k $K
    done ;;
  smat23)
    for D in 2 3; do for K in 1 2 3; do
      run "d$D k$K" --arm smat --d $D --k $K --read-mode plane --dirs oracle
    done; done ;;
  smat4)
    for K in 1 2 3; do
      run "d4 k$K" --arm smat --d 4 --k $K --read-mode plane --dirs oracle
    done ;;
  # the same masks with the hyperplane direction chosen by a learned map of the
  # query instead of handed over.  Softmax gets no oracle, so without these the
  # comparison is assisted on one side only.
  learned23)
    for D in 2 3; do for K in 1 2 3; do
      run "learned d$D k$K" --arm smat --d $D --k $K --read-mode plane --dirs learned
    done; done ;;
  learned4)
    for K in 1 2 3; do
      run "learned d4 k$K" --arm smat --d 4 --k $K --read-mode plane --dirs learned
    done ;;
  # the same, with one hidden layer in the direction head.  Fitting the oracle
  # direction directly, the linear head reaches 0.28 against a chance of 0.08
  # and this one reaches 1.00, so if the learned arm is still near zero here the
  # bottleneck is credit assignment through the mask rather than the head.
  # the proposed fix: score directions by the constraints they satisfy
  # (f[delta] summed over a local window) instead of by the token mixture, and
  # mix over directions during training instead of the straight-through argmax.
  # "soft" alone isolates the second from the first.
  fix3)
    run "delta d3 k2"       --arm smat --d 3 --k 2 --dirs learned --dir-head delta
    run "delta+soft d3 k2"  --arm smat --d 3 --k 2 --dirs learned --dir-head delta --dir-soft
    run "soft only d3 k2"   --arm smat --d 3 --k 2 --dirs learned --dir-soft ;;
  fix4)
    run "delta d4 k3"       --arm smat --d 4 --k 3 --dirs learned --dir-head delta
    run "delta+soft d4 k3"  --arm smat --d 4 --k 3 --dirs learned --dir-head delta --dir-soft ;;
  # the degenerate read: a type is a single cell rather than a hyperplane, i.e.
  # bucketing.  Table 5's comparison needs this under the same protocol as the
  # rest; the old point-read runs are batch 8 / lr 1e-3 like the other artefacts.
  point)
    for D in 3 4; do for K in 1 2 3; do
      run "point d$D k$K" --arm smat --d $D --k $K --read-mode point --dirs oracle
    done; done ;;
  # the decisive diagnostic: teach the direction head with an annealed
  # cross-entropy to the oracle, then remove it.  If the model keeps the
  # behaviour at coefficient zero, the map is learnable and reachable and only
  # the downstream signal was missing; if it does not, the obstacle is in the
  # read itself, not in the head.
  sup)
    run "sup delta d3 k2" --arm smat --d 3 --k 2 --dirs learned --dir-head delta \
        --dir-sup-coef 1.0 --dir-sup-anneal 6000
    run "sup delta d4 k3" --arm smat --d 4 --k 3 --dirs learned --dir-head delta \
        --dir-sup-coef 1.0 --dir-sup-anneal 6000 ;;
  # the direction computed, not learned and not supplied: hash the query's own
  # last w tokens with the mask's hash and take the direction annihilating their
  # differences.  Reads only the input, so it is a content-based type map in the
  # sense of Sec 3.2.  w = k-1 covers exactly the requested keys; w = 3 adds
  # noise tokens to the window and tests whether the rule survives them.
  fixed)
    run "fixed d3 k2 w1" --arm smat --d 3 --k 2 --dirs learned --dir-head fixed --dir-window 1
    run "fixed d4 k3 w2" --arm smat --d 4 --k 3 --dirs learned --dir-head fixed --dir-window 2
    run "fixed d3 k2 w3" --arm smat --d 3 --k 2 --dirs learned --dir-head fixed --dir-window 3 ;;
  # d=4 half of the point sweep, split out so it can run beside the d=3 half
  point4)
    for K in 1 2 3; do
      run "point d4 k$K" --arm smat --d 4 --k $K --read-mode point --dirs oracle
    done ;;
  # the rest of the table under the computed direction, so every SMAT cell comes
  # from a run that computes it rather than one that is handed it.  w = k-1 is
  # the window covering the requested keys; at k=1 the requested key is the
  # query's own token, whose cell lies on every hyperplane, so w is immaterial.
  fixed3)
    run "fixed d3 k1" --arm smat --d 3 --k 1 --dirs learned --dir-head fixed --dir-window 1
    run "fixed d3 k3" --arm smat --d 3 --k 3 --dirs learned --dir-head fixed --dir-window 2 ;;
  fixed4)
    run "fixed d4 k1" --arm smat --d 4 --k 1 --dirs learned --dir-head fixed --dir-window 1
    run "fixed d4 k2" --arm smat --d 4 --k 2 --dirs learned --dir-head fixed --dir-window 1 ;;
  mlpdir)
    run "mlp d3 k2" --arm smat --d 3 --k 2 --read-mode plane --dirs learned --dir-head mlp
    run "mlp d4 k3" --arm smat --d 4 --k 3 --read-mode plane --dirs learned --dir-head mlp
    run "mlp d2 k1" --arm smat --d 2 --k 1 --read-mode plane --dirs learned --dir-head mlp ;;
esac
echo "FAIR2 $GROUP DONE: $(date)"

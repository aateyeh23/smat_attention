#!/bin/bash
#SBATCH --job-name=smat-mkar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Multi-key subset recall: the first task whose difficulty scales in the thing d
# measures, and which also requires content addressing.  N pairs at random
# positions; a query names k keys and must emit a vector with support exactly
# the requested set.
#
# One row must admit exactly the k requested payloads.  In F_q^{d-1} a set of k
# points shares a hyperplane exactly when k <= d-1, verified directly against
# the direction table: coverage 1.000 for k <= d-1 and 0.06 / 0.24 above it.
# So the prediction is a staircase:
#
#          k=1    k=2    k=3
#   d=2    yes    no     no        (hyperplane = point)
#   d=3    yes    yes    no        (hyperplane = line)
#   d=4    yes    yes    yes       (hyperplane = plane)
#
# SINGLE LAYER, for the same reason tasks/routing_train.py is: with depth the k
# requested keys sit at separate recent positions, each fetches its own payload
# in one hop, and the answer row collects them through the recent causal block --
# so d never binds.  A 4-layer version scored 1.000 at d=2, k=2, which the
# geometry forbids for a single row.  Lemma "routing patterns are mask traces" is
# a per-layer statement and the task has to respect that.
#
# dirs=oracle hands the query a direction covering its k cells when one exists,
# so these arms measure what the MASK can do.  dirs=learned makes the model find
# the direction itself -- solving a small linear system over F_q, which I expect
# it to fail; the gap between the two is the learning problem, and without the
# split a failure says nothing about which of the two is responsible.
set -uo pipefail
source experiments/prelude.sh

BASE="--T 1024 --chunk 256 --n-pairs 8 --n-ans 8 --layers 1 --d-model 256 --heads 4 \
  --r 64 --nb 8 --lr 1e-3 --steps 4000 --eval-every 500 --seed 0 \
  --out results/mkar_T1024.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/mkar.py $BASE "${@:2}" || echo "FAILED: $1"; }

# the staircase, mask capability
for D in 2 3 4; do
  for K in 1 2 3; do
    run "d$D k$K oracle" --d $D --k $K --dirs oracle
  done
done
# the learning gap, where the mask is known to be capable
run "d3 k2 learned" --d 3 --k 2 --dirs learned
run "d4 k3 learned" --d 4 --k 3 --dirs learned
echo "MKAR DONE: $(date)"

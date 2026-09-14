#!/bin/bash
#SBATCH --job-name=smat-stream
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out

# Accuracy and cost in the SAME configuration, which the draft says is missing:
# every accuracy number comes from the content-addressed assignment and every
# cost exponent from the positional one, and the timing note in the cost section
# says outright that no timing claim should be made for the content-addressed
# path.  This job measures the decode claim -- O(rp) per token from a
# Theta(T^{1-1/d}) cache against O(T) from a Theta(T) one -- on the model that
# is also answering the task.
#
# Three measurements, kept separate on purpose:
#
#   A. cost, native geometry.  A fresh model at each length, so the cell count
#      is the one that length actually has.  Cost and cache do not depend on the
#      weights, so this runs to lengths nobody can afford to train at.  The
#      accuracy column is empty here and must stay empty.
#   B. accuracy and cost together, at the trained length.  The point of the job.
#   C. length extrapolation, frozen cells.  Trained at T=1024, read at longer
#      prefixes with the hash geometry held at its trained value.  The
#      assignment is content-based, so this is coherent to ask; whether it
#      survives is the experiment.  Labelled frozen-cells in the csv, and the
#      cache is constant there by construction -- do not read a scaling law off
#      those rows.
#
# Correctness gate: every mode runs the streaming decode against the full
# forward first and refuses to measure unless the logits agree.  On CPU at small
# shapes the relative error is 3.5e-07 with full argmax agreement, on all of
# smat d=1/2/3, point and plane reads, and softmax.
set -uo pipefail
source experiments/prelude.sh

CFG="--T 1024 --chunk 256 --layers 4 --d-model 256 --heads 4 --r 64 \
  --n-pairs 16 --n-queries 16 --n-keys 128 --short-conv 3 --hash-src id --hash-shift 1 \
  --read-mode plane --seed 0"
OUT="--out results/stream.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/streaming.py $CFG "${@:2}" || echo "FAILED: $1"; }

# -- gate ------------------------------------------------------------------
echo "### parity at the real shapes"
for D in 1 2 3; do
  run "parity d$D" --mode parity --d $D --nb 8 || true
done
run "parity softmax" --mode parity --arm softmax --d 1 --nb 8

# -- A: cost at native geometry -------------------------------------------
echo "### A: per-token cost and cache, native geometry"
for D in 2 3; do
  run "cost smat d$D" --mode cost --d $D --nb 4 --cost-steps 256 $OUT \
      --eval-T 1024 2048 4096 8192 16384 32768 65536
done
run "cost softmax" --mode cost --arm softmax --d 1 --nb 4 --cost-steps 256 $OUT \
    --eval-T 1024 2048 4096 8192 16384 32768 65536
run "cost smat d1" --mode cost --d 1 --nb 4 --cost-steps 256 $OUT \
    --eval-T 1024 2048 4096 8192 16384 32768 65536

# -- B + C: trained, then read at longer prefixes ---------------------------
echo "### B+C: accuracy and cost together, then extrapolation"
EV="--eval-T 1024 2048 4096 8192"
run "trained smat d2" --mode run --d 2 --nb 32 --steps 4000 --lr 1e-3 --eval-every 500 $EV $OUT --ckpt ckpt/stream_d2.pt
run "trained smat d3" --mode run --d 3 --nb 32 --steps 4000 --lr 1e-3 --eval-every 500 $EV $OUT --ckpt ckpt/stream_d3.pt
run "trained softmax" --mode run --arm softmax --d 1 --nb 32 --steps 4000 --lr 1e-3 --eval-every 500 $EV $OUT --ckpt ckpt/stream_sm.pt
run "trained smat d1" --mode run --d 1 --nb 32 --steps 4000 --lr 1e-3 --eval-every 500 $EV $OUT --ckpt ckpt/stream_d1.pt

echo "STREAM DONE: $(date)"

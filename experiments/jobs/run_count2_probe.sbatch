#!/bin/bash
#SBATCH --job-name=smat-cnt2p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# Is the rebuilt counting task solvable by ANY arm?  Two facts are now measured.
#
#   1. --short-conv 1 is a depthwise kernel of size 1 and mixes no neighbours,
#      so the answer row could not see the k requested symbols at t-k..t-1 and
#      the counted MARK could not see its symbol at p.  Fixed: width 4 reaches
#      t-3.  This was a real defect -- the caption in the draft claims the conv
#      is what rejects the 0.4-2.1 spurious admissions per row -- but it was NOT
#      what pinned the arms to the floor.
#   2. At width 4 the two arms with no oracle direction still sit at the exact
#      analytic marginal: k=1 loss 1.511 against a marginal entropy of 1.497
#      nats and accuracy 0.365 against a modal-class rate of 0.345; k=3 loss
#      1.869-2.030 against 1.933 and accuracy 0.208-0.248 against 0.228.  Both
#      arms predict the mode and read nothing from the sequence.
#
# For softmax that may be structural rather than a budget problem: attention
# values are query-independent and the read is normalised, so a row cannot turn
# "how many marks carry a requested symbol" into an average of a constant
# payload without holding its denominator fixed.  The d>=2 arms have a route
# the softmax arm does not -- the oracle direction admits the requested cells
# and the admitted mass is roughly constant at block/q -- and that route has
# never been run at width 4.
#
# So: the easiest on-diagonal cell and the hardest, at the settled budget
# (batch 32, lr 3e-4, 8000 steps).  If d=2 k=1 does not solve, the task is not
# well posed and the section should be cut rather than re-tuned.
set -uo pipefail
source experiments/prelude.sh

C="--T 1024 --chunk 256 --n-marks 12 --n-syms 512 --pool 8 --n-ans 8 --layers 1 \
  --d-model 256 --heads 4 --r 64 --nb 32 --lr 3e-4 --steps 8000 --eval-every 1000 \
  --short-conv 4 --seed 0 --out results/count2_probe.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u tasks/counting.py $C "${@:2}" || echo "FAILED: $1"; }

run "smat d2 k1" --arm smat --d 2 --k 1
run "smat d4 k3" --arm smat --d 4 --k 3
echo "PROBE DONE: $(date)"

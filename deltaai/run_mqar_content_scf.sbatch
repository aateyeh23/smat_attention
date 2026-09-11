#!/bin/bash
#SBATCH --job-name=smat-mqar-content
#SBATCH --partition=yugroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/mqar-content-%j.out

# MQAR core table, Berkeley SCF.  T=512, 16 pairs, 128 keys, under the gate
# recipe that fixes d=1 (dt init [0.1,1.0], no weight decay on the gates) --
# without it every arm sits at chance and the mask is not what is being tested.
#
#   softmax              control, should reach 1.0
#   smat d=1             linear attention; reference value 1.000 (DeltaAI, gate B)
#   smat d=2 / d=3       the published positional prof/type -- the control that
#                        was queued in run_lm_decay.sbatch and never ran
#   smat d=2/d=3 cpoint  content-addressed prof/type, own-cell read
#   smat d=3 cplane      content-addressed, hyperplane-of-a-hashed-direction read
#                        (at d=2 a hyperplane of F_q^1 is a point, so it coincides
#                         with cpoint and is not run separately)
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude_scf.sh"

COMMON="--decay mamba2 --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --qk-norm --short-conv 3 \
  --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 --layers 4 --d-model 256 --heads 4 --r 64 \
  --steps 5000 --nb 32 --lr 1e-3 --eval-every 1000 --eval-n 256 --early-stop 0.995 \
  --out results/mqar_T512_content_scf.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=8 python -u mqar_lm.py $COMMON "${@:2}" || echo "FAILED: $1"; }

for S in 0 1; do
  run "softmax s$S"             --arm softmax                                      --seed $S
  run "d1 s$S"                  --arm smat --d 1                                   --seed $S
  run "d2 positional s$S"       --arm smat --d 2                                   --seed $S
  run "d3 positional s$S"       --arm smat --d 3                                   --seed $S
  run "d2 content point s$S"    --arm smat --d 2 --assign content --read-mode point --seed $S
  run "d3 content point s$S"    --arm smat --d 3 --assign content --read-mode point --seed $S
  run "d3 content plane s$S"    --arm smat --d 3 --assign content --read-mode plane --seed $S
done
echo "SWEEP DONE: $(date)"

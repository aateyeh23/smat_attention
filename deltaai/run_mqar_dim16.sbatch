#!/bin/bash
#SBATCH --job-name=smat-dim16
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=01:30:00
#SBATCH --output=logs/dim16-%j.out

# Width 16, Zoology MQAR T=256, 4 pairs.  Real Mamba-2 mixer (mamba_ssm) as the
# reference arm in the same harness, against our gated d=1 with the signed
# (SSD-style, learned r=128) scan kernel.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
C="--zoology --T 256 --chunk 64 --layers 2 --d-model 16 --heads 1 --nb 64 --eval-every 1000 --eval-n 512 --early-stop 0.99 --n-pairs 4 --out results/mqar_dim16.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=4 python -u mqar_lm.py $C "${@:2}"; }
echo "==== pilot mamba2 ===="; OMP_NUM_THREADS=4 python -u mqar_lm.py --arm mamba2 $C --short-conv 0 --steps 20 --eval-every 20 --eval-n 64 --lr 1e-3 --out results/pilot_mqar.csv 2>&1 | grep -E "== eval|Error|Traceback|params"; rm -f results/pilot_mqar*
run "mamba2 lr1e-3 8k"       --arm mamba2 --short-conv 0 --lr 1e-3 --steps 8000  --tag _lr1e-3
run "mamba2 lr3e-3 8k"       --arm mamba2 --short-conv 0 --lr 3e-3 --steps 8000  --tag _lr3e-3
run "smat d1 signed lr1e-3 12k" --arm smat --d 1 --r 128 --scan-kernel signed --decay mamba2 --dt-min 0.05 --dt-max 0.5 --A-min 1 --A-max 2 --gate-wd 0 --qk-norm --short-conv 3 --lr 1e-3 --steps 12000 --tag _lr1e-3_12k
echo "done: $(date)"

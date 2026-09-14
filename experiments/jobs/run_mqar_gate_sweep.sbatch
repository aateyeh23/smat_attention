#!/bin/bash
#SBATCH --job-name=smat-gate-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Why doesn't the Mamba-2 gate learn to select on MQAR?  d=1, T=512, 16 pairs.
#   A  default init [1e-3,1e-1], no wd on gates, 5k steps        (isolates weight decay)
#   B  init [0.1,1.0], no wd on gates, 5k steps                   (responsive regime)
#   C  init [0.1,1.0], no wd, gate lr x5, 5k steps                (gate learning rate)
#   D  init [0.1,1.0], no wd, gate lr x5, 10k steps               (training length)
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
COMMON="--arm smat --d 1 --decay mamba2 --qk-norm --short-conv 3 --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 --layers 4 --d-model 256 --heads 4 --r 64 --nb 32 --lr 1e-3 --eval-every 1000 --eval-n 256 --out results/mqar_T512_gate_sweep.csv"
run() { echo "==== $1   $(date) ===="; OMP_NUM_THREADS=4 python -u tasks/mqar.py $COMMON "${@:2}"; }
run A --gate-wd 0 --steps 5000 --tag _A
run B --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --steps 5000 --tag _B
run C --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --gate-lr-mult 5 --steps 5000 --tag _C
run D --dt-min 0.1 --dt-max 1.0 --gate-wd 0 --gate-lr-mult 5 --steps 10000 --tag _D
echo "done: $(date)"

#!/bin/bash
#SBATCH --job-name=smat-mqar-mamba2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# MQAR T=512, 16 pairs, SMAT with Mamba-2 (SSD) scalar decay: dt=softplus, a=exp(-dt A), write scaled by dt in the causal
# scan (train_lm.decayed_segmented_scan; landmark pooling and mask unchanged).
# References (results/mqar_T512*.csv): softmax 100%, d=1 elu+1 1.3%, oracle 100%.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
for ARM in "--arm smat --d 1" "--arm smat --d 2"; do
    echo "==== arm: $ARM --decay mamba2   $(date) ===="
    OMP_NUM_THREADS=4 python -u tasks/mqar.py $ARM --decay mamba2 --qk-norm --short-conv 3 \
        --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 \
        --layers 4 --d-model 256 --heads 4 --r 64 \
        --steps 2500 --nb 32 --lr 1e-3 --eval-every 500 --eval-n 256 \
        --out results/mqar_T512_mamba2.csv
done
echo "done: $(date)"

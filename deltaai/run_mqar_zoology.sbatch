#!/bin/bash
#SBATCH --job-name=smat-mqar-zoo
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/mqar-zoo-%j.out

# Zoology MQAR (their generator, vocab 8192) at their four standard settings,
# 2-layer d_model 256 models (in their grid), causal short conv in every arm,
# QK-norm; SMAT arms with Mamba-2 style selective decay (dt init [0.05,0.5], A init [1,2] so initial decays lie in [0.37,0.95],
# no wd on gate params).  d=3 needs T >= 128.  Chunk = T/4.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
for CFG in "64 4 16" "128 8 32" "256 16 64" "512 64 128"; do
    set -- $CFG; T=$1; P=$2; C=$3
    for ARM in "--arm softmax" "--arm smat --d 1" "--arm smat --d 2" "--arm smat --d 3"; do
        [ "$T" = 64 ] && [ "$ARM" = "--arm smat --d 3" ] && continue
        echo "==== T=$T pairs=$P   arm: $ARM   $(date) ===="
        OMP_NUM_THREADS=4 python -u mqar_lm.py $ARM --zoology --T $T --n-pairs $P --chunk $C \
            --decay mamba2 --dt-min 0.05 --dt-max 0.5 --A-min 1 --A-max 2 --gate-wd 0 --qk-norm --short-conv 3 \
            --layers 2 --d-model 256 --heads 4 --r 64 \
            --steps 8000 --nb 64 --lr 1e-3 --eval-every 2000 --eval-n 512 \
            --out results/mqar_zoology_v2.csv
    done
done
echo "done: $(date)"

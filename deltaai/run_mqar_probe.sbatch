#!/bin/bash
#SBATCH --job-name=smat-mqar-probe
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:40:00
#SBATCH --output=logs/mqar-probe-%j.out

# State-pollution probe on MQAR (T=512, 16 pairs), SMAT d=1:
#   measurement: share of each query row's normaliser coming from noise tokens
#   intervention: --oracle-keep zeroes every noise token's key (no pollution)
# for the elu+1 and Taylor feature maps.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
for CFG in "--phi elu1" "--phi elu1 --oracle-keep" "--phi taylor --taylor-dim 16" "--phi taylor --taylor-dim 16 --oracle-keep"; do
    echo "==== arm: --arm smat --d 1 $CFG   $(date) ===="
    OMP_NUM_THREADS=4 python -u mqar_lm.py --arm smat --d 1 $CFG --qk-norm --short-conv 3 \
        --T 512 --n-pairs 16 --n-queries 16 --n-keys 128 \
        --layers 4 --d-model 256 --heads 4 --r 64 \
        --steps 2500 --nb 32 --lr 1e-3 --eval-every 500 --eval-n 256 \
        --out results/mqar_T512_probe.csv
done
echo "done: $(date)"

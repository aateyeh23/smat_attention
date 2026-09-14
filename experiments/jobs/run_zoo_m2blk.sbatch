#!/bin/bash
#SBATCH --job-name=zoo-m2blk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# SMAT arms inside Zoology's Mamba2Block (RMSNorm, no MLP, Mamba init path) -- the same
# block as the Mamba-2 reference.  Three sweeps share the GPU, lr 3.16e-3 then 1e-2:
#   A: d=1 (== Mamba-2, should reproduce the reference 85)   -> logs/zoo-m2blk-d1.out
#   B: plain hybrid d=2 (no reset, separate q/k/v for G)      -> logs/zoo-m2blk-hybrid-d2.out
#   C: multi-resolution d=2 (shared B/C/x, bounded lambda)    -> logs/zoo-m2blk-shared-d2.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_DM=16 ZOO_EPOCHS=32 ZOO_LRS=3.16e-3,1e-2
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
( ZOO_DS=1 ZOO_RESET=0 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/zoo-m2blk-d1.out 2>&1 ) &
( ZOO_DS=2 ZOO_RESET=0 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/zoo-m2blk-hybrid-d2.out 2>&1 ) &
( ZOO_DS=2 ZOO_RESET=0 ZOO_SHARED=1 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/zoo-m2blk-shared-d2.out 2>&1 ) &
wait
echo "done: $(date)"

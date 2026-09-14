#!/bin/bash
#SBATCH --job-name=zoo-pmask-d2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Paper mask + Mamba-2 weights + reset, width 16, seed 123: (a) d=2 both layers; (b) layer 0 d=1 (plain Mamba-2), layer 1 d=2.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_RESET_MR=1 ZOO_GDECAY=1 ZOO_LAMACT=one ZOO_POOL=sum ZOO_READ=type ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LRS=1e-2 ZOO_DM=16 ZOO_SEED=123
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
( ZOO_DS=2 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/zoo-pmask-d2-dm16-s123.out 2>&1 ) &
( ZOO_DS=2 ZOO_D_LAYERS=1-2 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/zoo-pmask-d1-2L-dm16-s123.out 2>&1 ) &
wait; echo "done: $(date)"

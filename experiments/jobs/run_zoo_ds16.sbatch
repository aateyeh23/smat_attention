#!/bin/bash
#SBATCH --job-name=zoo-ds16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Small-state sweep: headdim 16, d_state 16, d_model in {16, 32, 64}; Mamba-2 control (d=1) vs
# multi-res bounded-lambda SMAT (d=2), Mamba2Block, lr 1e-2, 32 epochs.  Six runs share the GPU.
#   -> logs/zoo-ds16-{ctrl,mr3}-dm{16,32,64}.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LAMACT=sigmoid ZOO_LRS=1e-2
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
for dm in 16 32 64; do
  ( ZOO_DM=$dm ZOO_DS=1 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/zoo-ds16-ctrl-dm$dm.out 2>&1 ) &
  ( ZOO_DM=$dm ZOO_DS=2 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/zoo-ds16-mr3-dm$dm.out 2>&1 ) &
done
wait
echo "done: $(date)"

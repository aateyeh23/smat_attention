#!/bin/bash
#SBATCH --job-name=hash-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Learned-hash training iteration: d=3 point read, width 16, one variant per GPU, each capped at 6 epochs / 5 min.
# VARIANTS: "name|ENV=.. ENV=.." lines, 4 per job.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-}
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_RESET_MR=1 ZOO_GDECAY=1 ZOO_LAMACT=one ZOO_POOL=sum ZOO_READ=chash ZOO_DS=3 ZOO_HCODIM=2 ZOO_HCONV=1 ZOO_BALANCE=0.01 ZOO_ANNEAL=2500 ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LRS=1e-2 ZOO_DM=16 ZOO_SEED=123
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
i=0
while IFS='|' read -r name envs; do [ -z "$name" ] && continue
  ( env CUDA_VISIBLE_DEVICES=$i $envs python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/hash-full-$name.out 2>&1; echo "ENDED $(date +%H:%M:%S)" >> logs/hash-full-$name.out ) &
  i=$((i+1)); sleep 2
done <<< "$VARIANTS"
echo "launched $i variants: $(date)"; wait; echo "done: $(date)"

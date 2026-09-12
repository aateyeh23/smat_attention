#!/bin/bash
#SBATCH --job-name=zoo-split
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/zoo-split-%j.out
# (A) d=3 c=2 frozen embedding hash  (B) d=4 c=3 frozen embedding hash  (C) d=4 per-head codim [1,2], learned hash + conv
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-}
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_RESET_MR=1 ZOO_GDECAY=1 ZOO_LAMACT=one ZOO_POOL=sum ZOO_READ=chash ZOO_HCONV=1 ZOO_BALANCE=0.01 ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LRS=1e-2 ZOO_DM=16 ZOO_SEED=123
ZOO_DS=4 ZOO_HCODIM=1,2 ZOO_ANNEAL=2500 python logs/smoke_split.py 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | tee logs/smoke_split.out
ZOO_DS=4 ZOO_HCODIM=3 ZOO_HSRC=embed ZOO_HFREEZE=1 python logs/smoke_split.py 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | tee -a logs/smoke_split.out
if [ "$(grep -c 'SMOKE OK' logs/smoke_split.out)" != 2 ]; then echo "SMOKE FAILED -- holding"; sleep 7000; exit 1; fi
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
( ZOO_DS=3 ZOO_HCODIM=2 ZOO_HSRC=embed ZOO_HFREEZE=1 python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-split-d3-c2-embfrz-dm16-s123.out 2>&1 ) &
( ZOO_DS=4 ZOO_HCODIM=3 ZOO_HSRC=embed ZOO_HFREEZE=1 python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-split-d4-c3-embfrz-dm16-s123.out 2>&1 ) &
( ZOO_DS=4 ZOO_HCODIM=1,2 ZOO_ANNEAL=2500 python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-split-d4-c12-dm16-s123.out 2>&1 ) &
wait; echo "done: $(date)"

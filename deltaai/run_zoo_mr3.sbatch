#!/bin/bash
#SBATCH --job-name=zoo-mr3
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/zoo-mr3-%j.out

# Multi-resolution rev 3 (SmatMamba2MR) vs a head-matched Mamba-2, width 16, Mamba2Block.
#   A: d=1 headdim 4 (== Mamba-2 with 8 heads)          lrs 1e-2, 3.16e-3   -> logs/zoo-mr3-d1-hd4.out
#   B: MR3 d=2 headdim 4 (bank covers all 13 profiles)   lrs 1e-2, 3.16e-3   -> logs/zoo-mr3-d2-hd4.out
#   C: MR3 d=2 headdim 4 + Mamba-2 decay in G, lr 1e-2; then headdim 8, lr 1e-2 -> logs/zoo-mr3-d2-variants.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_DM=16 ZOO_EPOCHS=32 ZOO_MR3=1
echo "==== smoke ===="
ZOO_DS=2 ZOO_HEADDIM=4 ZOO_LRS=1e-2 python logs/smoke_mr3.py 2>&1 | grep -v -i "warn\|hi mamba\|custom_"
if ! ZOO_DS=2 ZOO_HEADDIM=4 ZOO_LRS=1e-2 python logs/smoke_mr3.py 2>&1 | grep -q "SMOKE OK"; then
  echo "SMOKE FAILED -- holding allocation for an overlap fix"; sleep 7000; exit 1
fi
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
( ZOO_DS=1 ZOO_HEADDIM=4 ZOO_LRS=1e-2,3.16e-3 python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-mr3-d1-hd4.out 2>&1 ) &
( ZOO_DS=2 ZOO_HEADDIM=4 ZOO_LRS=1e-2,3.16e-3 python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-mr3-d2-hd4.out 2>&1 ) &
( ( ZOO_DS=2 ZOO_HEADDIM=4 ZOO_GDECAY=1 ZOO_LRS=1e-2 python -m zoology.launch zoo_smat_configs.py 2>&1;
    ZOO_DS=2 ZOO_HEADDIM=8 ZOO_LRS=1e-2 python -m zoology.launch zoo_smat_configs.py 2>&1 ) | eval "$FILT" > logs/zoo-mr3-d2-variants.out 2>&1 ) &
wait
echo "done: $(date)"

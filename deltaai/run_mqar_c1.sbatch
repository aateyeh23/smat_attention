#!/bin/bash
#SBATCH --job-name=mqarc1
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mqarc1-%j.out
# MQAR (Zoology, 2 layers, hd16/ds16, lr 1e-2, 32 epochs, seed 123) with the content hash actually TRAINED (Zoology's
# fit() now prebuilds the lazy hash modules before its optimizer; before 2026-09-12 the hash was a frozen random
# projection in every run).  SMAT d=2/3/4, hyperplane read (c=1), v10 recipe, widths 16/32/64; plus the d=3 point read
# at width 16 (the earlier 80-82 headline).  All runs concurrently on one GPU (2-6 GB each).
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-}
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_RESET_MR=1 ZOO_GDECAY=1 ZOO_LAMACT=one ZOO_POOL=sum ZOO_READ=chash ZOO_HCONV=1 ZOO_BALANCE=0.01 ZOO_BAL_GATED=1 ZOO_ANNEAL=1000 ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LRS=1e-2 ZOO_SEED=123
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "zoology\]|run_id=|valid/accuracy=|Traceback|Error" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
cd /u/archerdw/smat_attention/deltaai
for w in 16 32 64; do for d in 2 3 4; do
  ( env ZOO_DS=$d ZOO_HCODIM=1 ZOO_DM=$w ZOO_TAG=_c1w${w}_trained python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/mqar-c1t-w$w-d$d.out 2>&1; echo "ENDED $(date +%H:%M:%S)" >> logs/mqar-c1t-w$w-d$d.out ) &
  sleep 3
done; done
( env ZOO_DS=3 ZOO_HCODIM=2 ZOO_DM=16 ZOO_TAG=_ptw16_trained python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/mqar-pt-w16-d3.out 2>&1; echo "ENDED $(date +%H:%M:%S)" >> logs/mqar-pt-w16-d3.out ) &
wait; echo "ALL DONE $(date)"

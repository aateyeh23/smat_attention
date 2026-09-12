#!/bin/bash
#SBATCH --job-name=zoo-codim
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/zoo-codim-%j.out
# Affine-subspace family under content hashing: read = codim-c flat through the query's cell.
# Width 16, hd16/ds16, learned hash + learned key-side conv, anneal 2500, balance 0.01, seed 123.
# Existing: d=2 c=1 (86.1), d=3 c=1 (67.6).  New: d=3 c=2 | d=4 c=1 | d=4 c=2 | d=4 c=3   (T=256: d=4 -> q=5, 125 cells)
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_RESET_MR=1 ZOO_GDECAY=1 ZOO_LAMACT=one ZOO_POOL=sum ZOO_READ=chash ZOO_HCONV=1 ZOO_ANNEAL=2500 ZOO_BALANCE=0.01 ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LRS=1e-2 ZOO_DM=16 ZOO_SEED=123
NG=${NG:-4}
ZOO_DS=4 ZOO_HCODIM=2 python logs/smoke_codim.py 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | tee logs/smoke_codim.out
if ! grep -q "SMOKE OK" logs/smoke_codim.out; then echo "SMOKE FAILED -- holding"; sleep 7000; exit 1; fi
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
i=0
for cfg in "3 2" "4 1" "4 2" "4 3"; do set -- $cfg
  gpu=$((i % NG)); i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu ZOO_DS=$1 ZOO_HCODIM=$2 python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-codim-d$1-c$2-dm16-s123.out 2>&1 ) &
  sleep 2
done
echo "launched $i runs over $NG GPUs: $(date)"; wait; echo "done: $(date)"

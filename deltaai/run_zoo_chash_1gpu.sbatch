#!/bin/bash
#SBATCH --job-name=zoo-chash1
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/zoo-chash-%j.out
# Content-hashed SMAT (Abdullah's ContentAssign) inside the Zoology pipeline: paper mask C, Mamba-2 weights + reset,
# learned hash annealed soft->hard over 2500 steps, balance 0.01; width 16, hd16/ds16, lr 1e-2, seed 123.
#   d=2 point / d=3 plane  x  {fixed key shift 1, learned causal conv on the key-side hash input}
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_RESET_MR=1 ZOO_GDECAY=1 ZOO_LAMACT=one ZOO_POOL=sum ZOO_READ=chash ZOO_ANNEAL=2500 ZOO_BALANCE=0.01 ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LRS=1e-2 ZOO_DM=16 ZOO_SEED=123
NG=${NG:-4}
ZOO_DS=3 ZOO_HMODE=plane ZOO_HCONV=1 python logs/smoke_chash.py 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | tee logs/smoke_chash.out
if ! grep -q "SMOKE OK" logs/smoke_chash.out; then echo "SMOKE FAILED -- holding"; sleep 7000; exit 1; fi
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
i=0
for cfg in "2 point 0" "3 plane 0" "2 point 1" "3 plane 1"; do set -- $cfg
  gpu=$((i % NG)); i=$((i+1)); tag=$([ "$3" = 1 ] && echo hconv || echo sh1)
  ( CUDA_VISIBLE_DEVICES=$gpu ZOO_DS=$1 ZOO_HMODE=$2 ZOO_HCONV=$3 python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-chash-d$1-$2-$tag-dm16-s123.out 2>&1 ) &
  sleep 2
done
echo "launched $i runs over $NG GPUs: $(date)"; wait; echo "done: $(date)"

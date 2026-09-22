#!/bin/bash
#SBATCH --job-name=zoo-pmask1
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/zoo-pmask-%j.out

# The paper's mask M^(d) with Mamba-2 weights: identity kernel, no normaliser, dt-scaled writes, Mamba-2 decay
# (folded into G: key->boundary and boundary->query), recurrence RESET at the boundary (G is the only cross path),
# positional read through C with the mask bank, G added directly (per-head scale init 1).  d=1 == Mamba-2.
# hd16 / ds16, width $WIDTH, seed 123 x d in {1,2,3}, lr 1e-2, 32 epochs.  -> logs/zoo-pmask-d{d}-dm{W}-s{S}.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_RESET_MR=1 ZOO_GDECAY=1 ZOO_LAMACT=one ZOO_POOL=sum ZOO_READ=type ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LRS=1e-2
WIDTHS="${WIDTHS:-16 32}"
ZOO_DM=16 ZOO_DS=2 python logs/smoke_reset.py 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | tee logs/smoke_reset_dm16.out
if ! grep -q "SMOKE OK" logs/smoke_reset_dm16.out; then echo "SMOKE FAILED -- holding"; sleep 7000; exit 1; fi
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
i=0
for WIDTH in $WIDTHS; do for seed in 123; do for ds in 1 2 3; do
  gpu=0; i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu ZOO_DM=$WIDTH ZOO_DS=$ds ZOO_SEED=$seed python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-pmask-d$ds-dm$WIDTH-s$seed.out 2>&1 ) &
  sleep 2
done; done; done
echo "launched $i runs over 4 GPUs: $(date)"
wait
echo "done: $(date)"

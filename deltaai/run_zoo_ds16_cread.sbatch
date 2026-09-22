#!/bin/bash
#SBATCH --job-name=zoo-cread
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/zoo-cread-%j.out

# Option 1 + option 2 (delta-rule profile states + content-addressed read over all profiles) on the small-state sweep: headdim 16, d_state 16,
# d_model 16/32/64, multi-res bounded-lambda SMAT with ZOO_POOL=delta, lr 1e-2, 32 epochs.
# Baselines (Mamba-2 hd16/ds16: 46.9 / 82.8 / 81.5; sum-pool MR3: 40.8 / 71.7 / 82.4) already in logs/zoo-ds16-*.
#   -> logs/zoo-ds16-cread-dm{16,32,64}.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LAMACT=sigmoid ZOO_POOL=delta ZOO_READ=content ZOO_DS=2 ZOO_LRS=1e-2
echo "==== smoke ===="
ZOO_DM=16 python logs/smoke_cread.py 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | tee logs/smoke_cread.out
if ! grep -q "SMOKE OK" logs/smoke_cread.out; then echo "SMOKE FAILED -- holding allocation"; sleep 7000; exit 1; fi
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
for dm in 16 32 64; do
  ( ZOO_DM=$dm python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-ds16-cread-dm$dm.out 2>&1 ) &
done
wait
echo "done: $(date)"

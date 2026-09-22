#!/bin/bash
#SBATCH --job-name=selcopy1
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/selcopy-%j.out
# Selective copying (Mamba Sec 4.1.1): L 4096, 16 tokens, vocab 16; 2 layers, d_model 64, batch 64, lr 1e-3.
# Arms, one GPU each, resumable (each job trains up to --max_minutes then checkpoints):
#   mamba2 | smat d=2 reset | smat d=3 reset | smat d=3 no-reset (hybrid)
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/archerdw/smat_attention/deltaai
i=0
for cfg in "mamba2 1 1" "smat 2 1" "smat 3 1" "smat 3 0"; do set -- $cfg
  name=$1-d$2-r$3; ck=/work/hdd/bekw/archerdw/ckpt/sc-$name.pt
  ( CUDA_VISIBLE_DEVICES=0 python -u lm/sc_train.py --arm $1 --d $2 --reset $3 --ckpt $ck --max_minutes 110 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" >> logs/sc-$name.out 2>&1 ) &
  i=$((i+1)); sleep 3
done
wait; echo "JOB DONE $(date)"

#!/bin/bash
#SBATCH --job-name=sccal
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=logs/sccal-%j.out
# Calibrate the selective-copying difficulty on Mamba-2 alone: L = 4096, vocab 64, n_copy in {32, 64, 128, 256},
# 10k steps each, concurrently on one GPU (~6 min).  Pick the largest n_copy at which Mamba-2 still reaches >= 35%
# (target 40-50%), then launch run_sc.sbatch (all arms) at that setting.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/archerdw/smat_attention/deltaai
for nc in 32 64 128 256; do
  ( python -u lm/sc_train.py --arm mamba2 --d 1 --seq_len 4096 --n_copy $nc --n_vocab 64 --d_model 64 --n_layers 2 --headdim 64 --d_state 64 --batch 32 --steps 10000 --warmup 500 --lr 1e-3 --eval_every 500 --ckpt_every 500 --log_every 100 --max_minutes 35 --stop_acc 2 --ckpt /work/hdd/bekw/archerdw/ckpt/sccal-nc$nc.pt 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" > logs/sccal-nc$nc.out 2>&1 ) &
  sleep 2
done
wait
pick=""
for nc in 32 64 128 256; do
  acc=$(grep -E "^FINAL|^EVAL" logs/sccal-nc$nc.out | tail -1 | grep -o "acc [0-9.]*" | cut -d' ' -f2)
  best=$(grep -E "^EVAL" logs/sccal-nc$nc.out | grep -o "acc [0-9.]*" | cut -d' ' -f2 | sort -n | tail -1)
  echo "n_copy $nc: final acc $acc  best $best"
  [ -n "$acc" ] && awk "BEGIN{exit !($acc >= 0.35)}" && pick=$nc
done
if [ "$(grep -l "^FINAL" logs/sccal-nc*.out 2>/dev/null | wc -l)" -lt 4 ]; then echo "calibration incomplete (cancelled or timed out) -- not launching"; exit 1; fi
[ -z "$pick" ] && pick=32
echo "PICK n_copy=$pick (largest with Mamba-2 final acc >= 0.35)"
NC=$pick VOCAB=64 sbatch run_sc.sbatch && echo "launched run_sc.sbatch NC=$pick"
echo "CAL DONE $(date)"

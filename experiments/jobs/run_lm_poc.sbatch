#!/bin/bash
#SBATCH --job-name=lm-poc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# enwik8 bytes proof of concept: d_model 384, 6 layers, T 4096, batch 8, 1500 steps (49M bytes), lr 6e-4.
# Arms (one GPU each): mamba2 | smat d=2 | smat d=3 -- paper mask (reset at T/2), content hash with 16-byte key-side
# conv, v10 recipe (gated balance 0.01, anneal 500).  Then per-position loss on the 5M test bytes, window 501.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-}
cd /u/an author/smat_attention/the GPU cluster
COMMON="--bytes data/enwik8 --d_model 384 --n_layers 6 --headdim 64 --d_state 64 --seq_len 4096 --batch 8 --steps 1500 --warmup 100 --lr 6e-4 --anneal 500 --balance 0.01 --balance_gated 1 --hash_conv_width 16 --reset 1 --ckpt_every 250 --val_every 250 --log_every 25"
echo "=== smoke (synthetic, 3 steps) ==="
CUDA_VISIBLE_DEVICES=0 python -u tasks/lm.py --arm smat --d 3 --synthetic --steps 3 --log_every 1 --val_every 3 --ckpt_every 3 --ckpt /tmp/poc_smoke.pt --d_model 384 --n_layers 6 --seq_len 4096 --batch 2 --hash_conv_width 16 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | grep -E "arm=|step 3|DONE|Traceback|Error" | tee logs/lm-poc-smoke.out
if ! grep -q "TRAINING DONE" logs/lm-poc-smoke.out; then echo "SMOKE FAILED -- holding"; sleep 6000; exit 1; fi
rm -f /tmp/poc_smoke.pt
i=0
for cfg in "mamba2 1" "smat 2" "smat 3"; do set -- $cfg
  name=$1-d$2; ck=${SMAT_WORK}/ckpt/poc-$name.pt; rm -f $ck
  ( CUDA_VISIBLE_DEVICES=$i python -u tasks/lm.py --arm $1 --d $2 $COMMON --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" > logs/lm-poc-$name.out 2>&1
    CUDA_VISIBLE_DEVICES=$i python -u analysis/eval_pos_loss.py --ckpt $ck --out results/lm/poc-$name --split test --n_tokens 5e6 --seq_len 4096 --batch 8 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" >> logs/lm-poc-$name.out 2>&1 ) &
  i=$((i+1)); sleep 3
done
wait; echo "ALL DONE $(date)"

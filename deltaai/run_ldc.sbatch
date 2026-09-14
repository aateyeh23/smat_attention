#!/bin/bash
#SBATCH --job-name=ldc
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/ldc-%j.out
# Long-document LM: arxiv + PG-19 + pile long docs, GPT-2 BPE, 16K-token chunks in a fixed shared order.
# ARMS env: space-separated "arm:d[:codim]" items, one GPU each (e.g. "mamba2:1 smat:2 smat:3 smat:4" or "attn:0").
# codim defaults to 1 = the paper's hyperplane incidence; d-1 = point read (one cell per query).
# Identical model size / optimiser / schedule / batch / data order across arms; only the mixer differs.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/archerdw/smat_attention/deltaai
STEPS=${STEPS:-7630}          # 7630 x 4 x 16384 = 500M tokens
echo "=== smoke (LDC loader, hyperplane sparse path, 3 steps) ==="
CUDA_VISIBLE_DEVICES=0 python -u lm/train_lm2.py --arm smat --d 3 --hash_codim 1 --sparse_ops 1 --g_bf16 1 --ldc /work/hdd/bekw/archerdw/ldc --d_model 128 --n_layers 2 --seq_len 16384 --batch 2 --steps 3 --warmup 1 --anneal 2 --log_every 1 --val_every 3 --ckpt_every 3 --ckpt /tmp/ldc_smoke.pt 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | grep -E "arm=|step 3|val|DONE|Traceback|Error" | tee logs/ldc-smoke.out
if ! grep -q "TRAINING DONE" logs/ldc-smoke.out; then echo "SMOKE FAILED -- holding"; sleep 6000; exit 1; fi
rm -f /tmp/ldc_smoke.pt
COMMON="--ldc /work/hdd/bekw/archerdw/ldc --d_model ${DM:-768} --n_layers ${NL:-8} --headdim 64 --d_state 64 --seq_len 16384 --batch 4 --steps $STEPS --warmup 300 --lr 6e-4 --anneal 1000 --balance 0.01 --balance_gated 1 --hash_conv_width 4 --reset 1 --sparse_ops 1 --g_bf16 1 --ckpt_every 250 --val_every 250 --log_every 25"
i=0
for item in ${ARMS:-mamba2:1 smat:2 smat:3 smat:4}; do IFS=: read -r arm d c <<< "$item"; c=${c:-1}
  name=${TAG:-ldc}-$arm-d$d; [ "$arm" = smat ] && [ "$c" != 1 ] && name=$name-c$c; ck=/work/hdd/bekw/archerdw/ckpt/$name.pt
  ( CUDA_VISIBLE_DEVICES=$i python -u lm/train_lm2.py --arm $arm --d $d --hash_codim $c $COMMON --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" >> logs/$name.out 2>&1
    CUDA_VISIBLE_DEVICES=$i python -u lm/eval_pos_loss.py --ckpt $ck --out lm/out/$name --n_tokens 3.9e7 --seq_len 16384 --batch 4 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" >> logs/$name.out 2>&1 ) &
  i=$((i+1)); sleep 3
done
wait; echo "ALL DONE $(date)"

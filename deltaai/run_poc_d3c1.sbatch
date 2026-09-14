#!/bin/bash
#SBATCH --job-name=d3c1
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/d3c1-%j.out
# GPU0: enwik8 POC rerun, SMAT d=3 with the paper's HYPERPLANE read (c=1) on the exact sparse path; all other settings
#       identical to the point-read d=3 arm (v10 recipe, anneal 500, 16-byte key conv, fp32 G branch).
# GPU1-3: MQAR d=3 hyperplane read (ZOO_HCODIM=1) with the v10 / v9 / frozen-embedding hash recipes, 32 epochs --
#       the hyperplane read had only ever been run with the early unfixed recipe.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/archerdw/smat_attention/deltaai
COMMON="--bytes data/enwik8 --d_model 384 --n_layers 6 --headdim 64 --d_state 64 --seq_len 4096 --batch 8 --steps 1500 --warmup 100 --lr 6e-4 --anneal 500 --balance 0.01 --balance_gated 1 --hash_conv_width 16 --reset 1 --hash_codim 1 --sparse_ops 1 --ckpt_every 250 --val_every 250 --log_every 25"
echo "=== smoke (synthetic, 3 steps, hyperplane sparse path) ==="
CUDA_VISIBLE_DEVICES=0 python -u lm/train_lm2.py --arm smat --d 3 --synthetic --steps 3 --log_every 1 --val_every 3 --ckpt_every 3 --ckpt /tmp/d3c1_smoke.pt --d_model 384 --n_layers 6 --seq_len 4096 --batch 2 --hash_conv_width 16 --hash_codim 1 --sparse_ops 1 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | grep -E "arm=|step 3|DONE|Traceback|Error" | tee logs/d3c1-smoke.out
if ! grep -q "TRAINING DONE" logs/d3c1-smoke.out; then echo "SMOKE FAILED -- holding"; sleep 6000; exit 1; fi
rm -f /tmp/d3c1_smoke.pt
name=smat-d3-c1; ck=/work/hdd/bekw/archerdw/ckpt/poc-$name.pt; rm -f $ck
( CUDA_VISIBLE_DEVICES=0 python -u lm/train_lm2.py --arm smat --d 3 $COMMON --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" > logs/lm-poc-$name.out 2>&1
  CUDA_VISIBLE_DEVICES=0 python -u lm/eval_pos_loss.py --ckpt $ck --out lm/out/poc-$name --split test --n_tokens 5e6 --seq_len 4096 --batch 8 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" >> logs/lm-poc-$name.out 2>&1 ) &
# MQAR d=3 hyperplane arms
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_RESET_MR=1 ZOO_GDECAY=1 ZOO_LAMACT=one ZOO_POOL=sum ZOO_READ=chash ZOO_DS=3 ZOO_HCODIM=1 ZOO_HCONV=1 ZOO_BALANCE=0.01 ZOO_ANNEAL=2500 ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LRS=1e-2 ZOO_DM=16 ZOO_SEED=1
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
i=1
while IFS='|' read -r name envs; do [ -z "$name" ] && continue
  ( env CUDA_VISIBLE_DEVICES=$i $envs python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/hash-c1-$name.out 2>&1; echo "ENDED $(date +%H:%M:%S)" >> logs/hash-c1-$name.out ) &
  i=$((i+1)); sleep 3
done <<< "v10-c1|ZOO_BAL_GATED=1 ZOO_ANNEAL=1000 ZOO_TAG=_v10c1
v9-c1|ZOO_BALANCE=0 ZOO_HLR=0.1 ZOO_TAG=_v9c1
frozen-c1|ZOO_HSRC=embed ZOO_HFREEZE=1 ZOO_TAG=_frzc1"
wait; echo "ALL DONE $(date)"

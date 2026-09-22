#!/bin/bash
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/archerdw/smat_attention/deltaai
COMMON="--bytes data/enwik8 --d_model 384 --n_layers 6 --headdim 64 --d_state 64 --seq_len 4096 --batch 8 --steps 1500 --warmup 100 --lr 6e-4 --anneal 500 --balance 0.01 --balance_gated 1 --hash_conv_width 16 --reset 1 --ckpt_every 250 --val_every 250 --log_every 25"
for d in 2 3; do name=smat-d$d; ck=/work/hdd/bekw/archerdw/ckpt/poc-$name.pt; rm -f $ck
  ( CUDA_VISIBLE_DEVICES=0 python -u lm/train_lm2.py --arm smat --d $d $COMMON --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" > logs/lm-poc-$name.out 2>&1
    CUDA_VISIBLE_DEVICES=0 python -u lm/eval_pos_loss.py --ckpt $ck --out lm/out/poc-$name --split test --n_tokens 5e6 --seq_len 4096 --batch 8 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" >> logs/lm-poc-$name.out 2>&1 )
done

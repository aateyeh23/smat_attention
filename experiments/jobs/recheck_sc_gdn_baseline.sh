#!/bin/bash
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/an author/smat_attention/the GPU cluster
export PYTHONPATH="/u/an author/triton371:$PWD:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology" FLA_TILELANG=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled TRITON_F32_DEFAULT=tf32x3
export TRITON_CACHE_DIR="$PWD/results/mqar_gdn_reset/check_cache"
exec python -u tasks/copying_gdn.py --arm gdn --d 1 --reset 1 --d_model 64 --n_layers 2 --headdim 64 --d_state 16 --gdn_headdim 16 --seq_len 4096 --n_copy 16 --n_vocab 16 --batch 32 --steps 10000 --warmup 500 --lr 0.001 --wd 0.1 --anneal 1000 --balance 0.01 --eval_every 250 --ckpt_every 250 --seed 0 --hash_src ssm --lam_act sigmoid --ckpt results/sc_gdn_smat/check/reuse-d1.pt

#!/bin/bash
#SBATCH --job-name=lm-poc-d4
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/lm-poc-d4-%j.out
# enwik8 POC, same settings as the mamba2 / d=2 / d=3 arms, SMAT d=4 (T=4096: q=12, 1728 cells, point read c=3)
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/archerdw/smat_attention/deltaai
COMMON="--bytes data/enwik8 --d_model 384 --n_layers 6 --headdim 64 --d_state 64 --seq_len 4096 --batch 8 --steps 1500 --warmup 100 --lr 6e-4 --anneal 500 --balance 0.01 --balance_gated 1 --hash_conv_width 16 --reset 1 --ckpt_every 250 --val_every 250 --log_every 25"
ck=/work/hdd/bekw/archerdw/ckpt/poc-smat-d4.pt; rm -f $ck
python -u lm/train_lm2.py --arm smat --d 4 $COMMON --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" > logs/lm-poc-smat-d4.out 2>&1
python -u lm/eval_pos_loss.py --ckpt $ck --out lm/out/poc-smat-d4 --split test --n_tokens 5e6 --seq_len 4096 --batch 8 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" >> logs/lm-poc-smat-d4.out 2>&1
echo "ALL DONE $(date)"

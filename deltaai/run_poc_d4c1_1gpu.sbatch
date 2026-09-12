#!/bin/bash
#SBATCH --job-name=d4c1i
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/d4c1i-%j.out
# 1-GPU interactive hedge for the enwik8 d=4 hyperplane arm of run_poc_d3c1.sbatch: same run, separate ckpt/log name (-c1b).
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/archerdw/smat_attention/deltaai
COMMON="--bytes data/enwik8 --d_model 384 --n_layers 6 --headdim 64 --d_state 64 --seq_len 4096 --batch 8 --steps 1500 --warmup 100 --lr 6e-4 --anneal 500 --balance 0.01 --balance_gated 1 --hash_conv_width 16 --reset 1 --hash_codim 1 --sparse_ops 1 --hash_ckpt 1 --ckpt_every 250 --val_every 250 --log_every 25"
name=smat-d4-c1i; ck=/work/hdd/bekw/archerdw/ckpt/poc-$name.pt; rm -f $ck
python -u lm/train_lm2.py --arm smat --d 4 $COMMON --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" > logs/lm-poc-$name.out 2>&1
python -u lm/eval_pos_loss.py --ckpt $ck --out lm/out/poc-$name --split test --n_tokens 5e6 --seq_len 4096 --batch 8 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" >> logs/lm-poc-$name.out 2>&1
echo "ALL DONE $(date)"

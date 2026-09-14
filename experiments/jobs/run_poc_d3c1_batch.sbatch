#!/bin/bash
#SBATCH --job-name=d3c1b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Backfill hedge for the enwik8 d=3 hyperplane arm of run_poc_d3c1.sbatch: same run, separate ckpt/log name (-c1b).
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/an author/smat_attention/the GPU cluster
COMMON="--bytes data/enwik8 --d_model 384 --n_layers 6 --headdim 64 --d_state 64 --seq_len 4096 --batch 8 --steps 1500 --warmup 100 --lr 6e-4 --anneal 500 --balance 0.01 --balance_gated 1 --hash_conv_width 16 --reset 1 --hash_codim 1 --sparse_ops 1 --ckpt_every 250 --val_every 250 --log_every 25"
name=smat-d3-c1b; ck=/work/hdd/bekw/an author/ckpt/poc-$name.pt; rm -f $ck
python -u tasks/lm.py --arm smat --d 3 $COMMON --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" > logs/lm-poc-$name.out 2>&1
python -u analysis/eval_pos_loss.py --ckpt $ck --out results/lm/poc-$name --split test --n_tokens 5e6 --seq_len 4096 --batch 8 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" >> logs/lm-poc-$name.out 2>&1
echo "ALL DONE $(date)"

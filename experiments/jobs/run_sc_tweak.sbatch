#!/bin/bash
#SBATCH --job-name=sctweak
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Selective copying, no-reset SMAT, optimisation tweaks aimed at the ordinal-addressing solution:
#   gate : learned per-head sigmoid gate on the G read (init off, lam_bias -8) instead of fixed weight 1
#   ssm  : hash the SSM output y_t (carries counts / context in every layer) instead of the layer input
#   nobal: no balance penalty, anneal 2000 (let the hash organise itself around the ordinals)
# d_state 16, L 4096, 16 tokens / vocab 16, batch 32, 10k steps; baselines Mamba-2 77.1, GDN 96.0, plain no-reset
# SMAT d=3 34.9 / d=4 65.0.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/an author/smat_attention/the GPU cluster
BASE="--arm smat --reset 0 --seq_len 4096 --n_copy 16 --n_vocab 16 --d_model 64 --n_layers 2 --headdim 64 --d_state 16 --batch 32 --steps 10000 --warmup 500 --lr 1e-3 --hash_conv_width 4 --eval_every 250 --ckpt_every 250 --log_every 50 --max_minutes 112"
echo "=== smoke"; python -u tasks/copying.py $BASE --d 3 --lam_act sigmoid --hash_src ssm --steps 3 --log_every 1 --eval_every 3 --ckpt_every 3 --max_minutes 5 --ckpt /tmp/sctweak_smoke.pt 2>&1 | grep -v -i "warn\|hi mamba\|custom_\|torchao" | grep -E "optimizer covers|FINAL|Traceback|Error" | tee logs/sctweak-smoke.out; rm -f /tmp/sctweak_smoke.pt
grep -q FINAL logs/sctweak-smoke.out || { echo "SMOKE FAILED"; exit 1; }
while IFS='|' read -r name extra; do [ -z "$name" ] && continue
  ( python -u tasks/copying.py $BASE $extra --ckpt /work/hdd/bekw/an author/ckpt/sc4k-tweak-$name.pt 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" > logs/sc4k-tweak-$name.out 2>&1
    echo "END $name: $(grep -E 'SOLVED|FINAL|TIME LIMIT|Traceback' logs/sc4k-tweak-$name.out | tail -1 | cut -c1-80)" ) &
  sleep 2
done <<'ARMS'
d3-gate|--d 3 --lam_act sigmoid --anneal 1000 --balance 0.01 --balance_gated 1
d3-gate-ssm|--d 3 --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1
d3-ssm|--d 3 --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1
d3-gate-ssm-nobal|--d 3 --lam_act sigmoid --hash_src ssm --anneal 2000 --balance 0
d4-gate-ssm|--d 4 --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1
ARMS
wait; echo "ALL DONE $(date)"

#!/bin/bash
#SBATCH --job-name=fixd4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# (1) resume the PG-19 16K smat d=4 arm with UNFILTERED stderr (it was dying silently ~35 s after resume, looping the
#     chain 190 times); eval if it finishes.  (2) the selective-copying seed runs, concurrently.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/an author/smat_attention/the GPU cluster
SEQ=16384; BATCH=2; STEPS=9155
COMMON="--data ${SMAT_WORK}/pg19 --d_model 384 --n_layers 8 --headdim 64 --d_state 64 --seq_len $SEQ --batch $BATCH --steps $STEPS --warmup 300 --lr 6e-4 --anneal 1000 --balance 0.01 --balance_gated 1 --hash_conv_width 4 --reset 1 --hash_codim 1 --sparse_ops 1 --hash_ckpt 1 --ckpt_every 250 --val_every 500 --log_every 25"
name=pg19-16k-smat-d4; ck=${SMAT_WORK}/ckpt/$name.pt
( python -u tasks/lm.py --arm smat --d 4 $COMMON --ckpt $ck > logs/lm-$name.debug.out 2>&1; echo "train exit code $?" >> logs/lm-$name.debug.out
  grep -E "^step|VAL|resumed|TRAINING DONE" logs/lm-$name.debug.out >> logs/lm-$name.out
  if grep -q "TRAINING DONE" logs/lm-$name.debug.out; then
    python -u analysis/eval_pos_loss.py --ckpt $ck --out results/lm/$name --split test --n_tokens 5e6 --seq_len $SEQ --batch $BATCH 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" >> logs/lm-$name.out 2>&1
    echo "DONE $name: $(grep -o '"bpb": [0-9.]*' logs/lm-$name.out | tail -1)"
  fi ) &
sleep 20
BASE="--arm smat --reset 0 --seq_len 4096 --n_copy 16 --n_vocab 16 --d_model 64 --n_layers 2 --headdim 64 --d_state 16 --batch 32 --steps 10000 --warmup 500 --lr 1e-3 --hash_conv_width 4 --eval_every 250 --ckpt_every 250 --log_every 50 --max_minutes 110"
launch() { name=$1; extra=$2; grep -q "FINAL\|SOLVED" logs/sc4k-tweak-$name.out 2>/dev/null && return; E2=""; case "$extra" in *"--arm gdn"*) E2="PYTHONPATH=/u/an author/triton371:$PYTHONPATH FLA_TILELANG=0";; esac
  ( env $E2 python -u tasks/copying.py $BASE $extra --ckpt ${SMAT_WORK}/ckpt/sc4k-tweak-$name.pt 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" >> logs/sc4k-tweak-$name.out 2>&1
    echo "END $name: $(grep -E 'SOLVED|FINAL|TIME LIMIT|Traceback' logs/sc4k-tweak-$name.out | tail -1 | cut -c1-80)" ) &
  sleep 2; }
launch d3-gate-s1 "--d 3 --lam_act sigmoid --anneal 1000 --balance 0.01 --balance_gated 1 --seed 1"
launch d3-gate-s2 "--d 3 --lam_act sigmoid --anneal 1000 --balance 0.01 --balance_gated 1 --seed 2"
launch d4-gate-ssm-s1 "--d 4 --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1 --seed 1"
launch d4-gate-ssm-s2 "--d 4 --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1 --seed 2"
launch gdn-s1 "--arm gdn --gdn_headdim 16 --seed 1"
launch gdn-s2 "--arm gdn --gdn_headdim 16 --seed 2"
wait; echo "ALL DONE $(date)"

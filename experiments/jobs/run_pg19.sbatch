#!/bin/bash
#SBATCH --job-name=pg19
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# PG-19 (GPT-2 BPE) sweep: 8 layers, d_model 384 (33M params incl. 19M tied embedding), 300M training tokens per arm
# (9155 steps x 32K tokens), SEQ in {16384, 32768}; arms mamba2, gated deltanet (fla), transformer (attn + 4x MLP, 14.2M non-emb ~ SMAT 13.9M), smat d=2/3/4 hyperplane read.  All four arms of one
# SEQ share the GPU concurrently (the SMAT layer is launch-bound, so this beats running them back to back).  Resumable:
# checkpoints every 250 steps, arms with a finished eval are skipped, and the job resubmits itself until everything is
# done.  Per-position loss on the PG-19 test split (100 books) at the training length.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/an author/smat_attention/the GPU cluster
T0=$(date +%s); BUDGET=$((105 * 60))                       # seconds of training per job before checkpoint-and-resubmit
STEPS=${STEPS:-9155}
for SEQ in ${SEQS:-16384 32768}; do
  BATCH=$((32768 / SEQ)); TAG=pg19-$((SEQ / 1024))k
  COMMON="--data /work/hdd/bekw/an author/pg19 --d_model 384 --n_layers 8 --headdim 64 --d_state 64 --seq_len $SEQ --batch $BATCH --steps $STEPS --warmup 300 --lr 6e-4 --anneal 1000 --balance 0.01 --balance_gated 1 --hash_conv_width 4 --reset 1 --hash_codim 1 --sparse_ops 1 --hash_ckpt 1 --ckpt_every 250 --val_every 500 --log_every 25"
  todo=""
  for cfg in ${ARMS:-"mamba2 1" "gdn 1" "attn 0" "smat 2" "smat 3" "smat 4"}; do set -- $cfg
    name=$TAG-$1-d$2; [ -f results/lm/$name.json ] && continue; todo="$todo|$cfg"
  done
  [ -z "$todo" ] && continue
  left=$((BUDGET - ($(date +%s) - T0))); [ $left -lt 300 ] && break
  MAXC=$((SEQ <= 16384 ? 2 : 1)); running=0                # arms sharing the GPU at once (~15 GB each at 16K, ~30 GB at 32K)
  echo "=== $TAG arms:$todo  budget ${left}s  concurrency $MAXC  $(date +%H:%M:%S)"
  IFS='|'; for cfg in $todo; do [ -z "$cfg" ] && continue; IFS=' '; set -- $cfg
    name=$TAG-$1-d$2; ck=/work/hdd/bekw/an author/ckpt/$name.pt
    left=$((BUDGET - ($(date +%s) - T0))); [ $left -lt 300 ] && break
    ( ENV=""; [ "$1" = gdn ] && ENV="env PYTHONPATH=/u/an author/triton371:$PYTHONPATH FLA_TILELANG=0"
      timeout $left $ENV python -u tasks/lm.py --arm $1 --d $2 $COMMON --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" >> logs/lm-$name.out 2>&1
      if grep -q "TRAINING DONE" logs/lm-$name.out; then
        $ENV python -u analysis/eval_pos_loss.py --ckpt $ck --out results/lm/$name --split test --n_tokens 5e6 --seq_len $SEQ --batch $BATCH 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" >> logs/lm-$name.out 2>&1
        echo "DONE $name: $(grep -o '"bpb": [0-9.]*' logs/lm-$name.out | tail -1)"
      fi ) &
    running=$((running + 1)); sleep 5
    if [ $running -ge $MAXC ]; then wait -n; running=$((running - 1)); fi
    IFS='|'
  done; IFS=' '
  wait
done
if ls results/lm/pg19-*.json 2>/dev/null | wc -l | grep -q "^12$"; then echo "ALL DONE $(date)"; else echo "resubmitting $(date)"; for i in $(seq 1 30); do sbatch run_pg19.sbatch && break; echo "submit refused (QOS limit), retry $i in 60s"; sleep 60; done; fi

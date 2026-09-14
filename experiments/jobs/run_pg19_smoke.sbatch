#!/bin/bash
#SBATCH --job-name=pg19smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
# Smoke for run_pg19.sbatch: each arm alone on one GPU at SEQ 32768 (synthetic tokens, vocab 50257), 40 steps ->
# per-arm throughput and peak memory, to decide how many arms can share a GPU; also checks the GDN / transformer arms.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/an author/smat_attention/the GPU cluster
COMMON="--synthetic --d_model 384 --n_layers 8 --headdim 64 --d_state 64 --seq_len 32768 --batch 1 --steps 40 --warmup 5 --anneal 20 --balance 0.01 --balance_gated 1 --hash_conv_width 4 --reset 1 --hash_codim 1 --sparse_ops 1 --hash_ckpt 1 --ckpt_every 40 --val_every 40 --log_every 10"
for cfg in "mamba2 1" "gdn 1" "attn 0" "smat 2" "smat 3" "smat 4"; do set -- $cfg
  ENV=""; [ "$1" = gdn ] && ENV="env PYTHONPATH=/u/an author/triton371:$PYTHONPATH FLA_TILELANG=0"
  $ENV python -u tasks/lm.py --arm $1 --d $2 $COMMON --ckpt /tmp/pg19smoke-$1-$2.pt 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" | grep --line-buffered -E "arm=|step (10|20|30|40) |Traceback|Error|OutOfMemory" | sed -u "s/^/[$1 d=$2] /"
done
rm -f /tmp/pg19smoke-*.pt; echo "SMOKE DONE"

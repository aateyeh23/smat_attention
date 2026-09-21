#!/bin/bash
#SBATCH --job-name=poclen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# enwik8 POC at long context: SEQ env (16384 | 32768), tokens/step held at 32K (batch = 32768/SEQ), 1500 steps = 49M
# bytes, same model / recipe as the T=4096 arms.  Arms run one after another on one GPU: mamba2, smat d=2/3/4 with the
# paper's hyperplane read (c=1).  Per-position loss on the 5M test bytes at the training length.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/an author/smat_attention/the GPU cluster
SEQ=${SEQ:-16384}; BATCH=$((32768 / SEQ)); TAG=poc$((SEQ / 1024))k
COMMON="--bytes data/enwik8 --d_model 384 --n_layers 6 --headdim 64 --d_state 64 --seq_len $SEQ --batch $BATCH --steps 1500 --warmup 100 --lr 6e-4 --anneal 500 --balance 0.01 --balance_gated 1 --hash_conv_width 16 --reset 1 --hash_codim 1 --sparse_ops 1 --hash_ckpt 1 --ckpt_every 250 --val_every 250 --log_every 25"
for cfg in ${ARMS:-"mamba2 1" "smat 2" "smat 3" "smat 4"}; do set -- $cfg
  name=$TAG-$1-d$2; ck=${SMAT_WORK}/ckpt/$name.pt; rm -f $ck
  echo "=== $name $(date +%H:%M:%S)"
  python -u tasks/lm.py --arm $1 --d $2 $COMMON --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" > logs/lm-$name.out 2>&1
  python -u analysis/eval_pos_loss.py --ckpt $ck --out results/lm/$name --split test --n_tokens 5e6 --seq_len $SEQ --batch $BATCH 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" >> logs/lm-$name.out 2>&1
  grep -E "bpb|Traceback|Error" logs/lm-$name.out | tail -2 | cut -c1-200
done
echo "ALL DONE $(date)"

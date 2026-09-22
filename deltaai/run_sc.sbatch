#!/bin/bash
#SBATCH --job-name=selcopy
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/selcopy-%j.out
# Selective copying (Mamba paper Sec. 4.1.1 form): L = 4096, NC content tokens (vocab VOCAB) at random positions
# among noise, 16 answer slots at the end; 2 layers, d_model 64, head_dim 64, d_state 64.  Arms run concurrently on
# one GPU (tiny models): mamba2, gated deltanet, smat d=2/3/4 with the hyperplane read and the boundary reset at 2048
# (the first half of the prefix is reachable only through the hashed block G).  Batch 32, up to 10k steps, eval on 256
# fixed sequences every 250 steps, stops once accuracy >= 99.5% twice in a row.  Resumable.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/archerdw/smat_attention/deltaai
L=${L:-4096}; NC=${NC:-16}; VOCAB=${VOCAB:-16}; DS=${DS:-64}; GDNHD=${GDNHD:-64}
COMMON="--seq_len $L --n_copy $NC --n_vocab $VOCAB --d_model 64 --n_layers 2 --headdim 64 --d_state $DS --gdn_headdim $GDNHD --batch 32 --steps 10000 --warmup 500 --lr 1e-3 --anneal 1000 --balance 0.01 --balance_gated 1 --hash_conv_width 4 --eval_every 250 --ckpt_every 250 --log_every 50 --max_minutes 112"
for cfg in ${ARMS:-"mamba2 1" "gdn 1" "smat 2" "smat 3" "smat 4"}; do set -- $cfg
  name=sc$((L / 1024))k-nc$NC-ds$DS-$1-d$2; ck=/work/hdd/bekw/archerdw/ckpt/$name.pt
  ENV=""; [ "$1" = gdn ] && ENV="env PYTHONPATH=/u/archerdw/triton371:$PYTHONPATH FLA_TILELANG=0"
  ( $ENV python -u lm/sc_train.py --arm $1 --d $2 $COMMON --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" >> logs/$name.out 2>&1
    echo "END $name: $(grep -E 'SOLVED|FINAL|TIME LIMIT|Traceback' logs/$name.out | tail -1 | cut -c1-120)" ) &
  sleep 2
done
wait; echo "ALL DONE $(date)"

#!/bin/bash
#SBATCH --job-name=pg19m9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Parameter-matched Mamba-2 control for the PG-19 table: 9 layers (8.5M non-emb, vs SMAT's ~8.6M effective = 7.55M base
# + ~1.0M used hash/direction params), both lengths concurrently.  Same data order, schedule and budget as the sweep.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /u/an author/smat_attention/the GPU cluster
for SEQ in 16384 32768; do BATCH=$((32768 / SEQ)); name=pg19-$((SEQ / 1024))k-mamba2L9-d1; ck=/work/hdd/bekw/an author/ckpt/$name.pt
  ( python -u tasks/lm.py --arm mamba2 --d 1 --data /work/hdd/bekw/an author/pg19 --d_model 384 --n_layers 9 --headdim 64 --d_state 64 --seq_len $SEQ --batch $BATCH --steps 9155 --warmup 300 --lr 6e-4 --reset 1 --ckpt_every 250 --val_every 500 --log_every 25 --ckpt $ck 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" > logs/lm-$name.out 2>&1
    python -u analysis/eval_pos_loss.py --ckpt $ck --out results/lm/$name --split test --n_tokens 5e6 --seq_len $SEQ --batch $BATCH 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_\|torchao" >> logs/lm-$name.out 2>&1
    echo "DONE $name: $(grep -o '"bpb": [0-9.]*' logs/lm-$name.out | tail -1)" ) &
  sleep 5
done
wait; echo "ALL DONE $(date)"

#!/bin/bash
#SBATCH --job-name=smat-strd3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# The d=3 trained arm of run_stream.sbatch, which ran out of memory on an
# 11 GB 2080 Ti: the content-addressed pool stacks one (r, p) state per cell and
# d=3 has many more cells than d=2, so batch 32 does not fit.  Everything else
# in that job is unaffected -- the d=3 cost sweep ran at batch 4 and is in the
# csv already.  The 11-16 GB nodes are excluded here rather than dropping the
# batch, because batch 32 is the settled budget and an arm trained at batch 8
# would not be comparable to the rest of the table.
set -uo pipefail
source experiments/prelude.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv

CFG="--T 1024 --chunk 256 --layers 4 --d-model 256 --heads 4 --r 64 \
  --n-pairs 16 --n-queries 16 --n-keys 128 --short-conv 3 --hash-src id --hash-shift 1 \
  --read-mode plane --seed 0"
echo "==== trained smat d3   $(date) ===="
OMP_NUM_THREADS=8 python -u tasks/streaming.py $CFG --mode run --d 3 --nb 32 --steps 4000 \
  --lr 1e-3 --eval-every 500 --eval-T 1024 2048 4096 8192 \
  --out results/stream.csv --ckpt ckpt/stream_d3.pt || echo "FAILED: trained smat d3"
echo "STREAM D3 DONE: $(date)"

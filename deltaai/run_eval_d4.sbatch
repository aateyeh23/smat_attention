#!/bin/bash
#SBATCH --job-name=eval-d4
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=logs/eval-d4-%j.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-}
cd /u/archerdw/smat_attention/deltaai
python -u lm/eval_pos_loss.py --ckpt /work/hdd/bekw/archerdw/ckpt/poc-smat-d4.pt --out lm/out/poc-smat-d4 --split test --n_tokens 5e6 --seq_len 4096 --batch 8 2>&1 | grep --line-buffered -v -i "warn\|hi mamba\|custom_" | tee -a logs/lm-poc-smat-d4.out

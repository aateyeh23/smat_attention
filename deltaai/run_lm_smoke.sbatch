#!/bin/bash
#SBATCH --job-name=lm-smoke
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --output=logs/lm-smoke-%j.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-}
cd /u/archerdw/smat_attention/deltaai
echo "=== mixer benchmark, T=16384 ==="; python lm/bench_mixer.py 2>&1 | grep -v -i "warn\|hi mamba\|custom_"
for arm in mamba2 smat; do
  echo "=== trainer smoke: $arm (synthetic, 20 steps) ==="
  rm -f /tmp/lm_smoke_$arm.pt
  python lm/train_lm2.py --arm $arm --synthetic --steps 20 --log_every 5 --val_every 20 --ckpt_every 20 --ckpt /tmp/lm_smoke_$arm.pt 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | grep -E "arm=|step|VAL|DONE|Error|Traceback" 
done
echo "SMOKE DONE"

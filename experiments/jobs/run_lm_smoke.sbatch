#!/bin/bash
#SBATCH --job-name=lm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-}
cd /u/an author/smat_attention/the GPU cluster
echo "=== mixer benchmark, T=16384 ==="; python analysis/bench_mixer.py 2>&1 | grep -v -i "warn\|hi mamba\|custom_"
for arm in mamba2 smat; do
  echo "=== trainer smoke: $arm (synthetic, 20 steps) ==="
  rm -f /tmp/lm_smoke_$arm.pt
  python tasks/lm.py --arm $arm --synthetic --steps 20 --log_every 5 --val_every 20 --ckpt_every 20 --ckpt /tmp/lm_smoke_$arm.pt 2>&1 | grep -v -i "warn\|hi mamba\|custom_" | grep -E "arm=|step|VAL|DONE|Error|Traceback" 
done
echo "SMOKE DONE"

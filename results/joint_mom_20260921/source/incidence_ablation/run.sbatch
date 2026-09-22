#!/bin/bash
#SBATCH --job-name=joint-incidence
#SBATCH --account=<account>
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=results/joint_incidence_20260921/logs/train-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd .
export PYTHONSAFEPATH=1
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/incidence_ablation:<triton-3.7.1>:$PWD/results/joint_incidence_20260921/source/the GPU cluster:$PWD/results/joint_incidence_20260921/source/smat:$PWD/results/joint_incidence_20260921/source:<site-packages>"
export TRITON_CACHE_DIR="$PWD/results/synthetic_memory/triton_cache"
python -P -u results/joint_incidence_20260921/source/joint_recall_campaign.py --manifest results/joint_incidence_20260921/tasks.json --minutes 345
python -P -u incidence_ablation/collect.py
if python - <<'PY'
import json
from pathlib import Path
root=Path('results/joint_incidence_20260921')
tasks=json.loads((root/'tasks.json').read_text())
raise SystemExit(0 if all(Path(t['result']).exists() for t in tasks) else 1)
PY
then exit 0; fi
INCIDENCE_SLICE=${INCIDENCE_SLICE:-1}
if [ "$INCIDENCE_SLICE" -ge 3 ]; then exit 2; fi
INCIDENCE_NEXT=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" \
    --export="ALL,INCIDENCE_SLICE=$((INCIDENCE_SLICE+1))" incidence_ablation/run.sbatch)
printf '%s\n' "$INCIDENCE_NEXT" >> results/joint_incidence_20260921/jobs.txt

#!/bin/bash
#SBATCH --job-name=joint-incidence
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/results/joint_incidence_20260921/logs/train-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention
export PYTHONSAFEPATH=1
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/deltaai/incidence_ablation:/u/archerdw/triton371:$PWD/deltaai/results/joint_incidence_20260921/source/deltaai:$PWD/deltaai/results/joint_incidence_20260921/source/smat:$PWD/deltaai/results/joint_incidence_20260921/source:/u/archerdw/.local/lib/python3.12/site-packages"
export TRITON_CACHE_DIR="$PWD/deltaai/results/synthetic_memory/triton_cache"
python -P -u deltaai/results/joint_incidence_20260921/source/deltaai/joint_recall_campaign.py --manifest deltaai/results/joint_incidence_20260921/tasks.json --minutes 345
python -P -u deltaai/incidence_ablation/collect.py
if python - <<'PY'
import json
from pathlib import Path
root=Path('deltaai/results/joint_incidence_20260921')
tasks=json.loads((root/'tasks.json').read_text())
raise SystemExit(0 if all(Path(t['result']).exists() for t in tasks) else 1)
PY
then exit 0; fi
INCIDENCE_SLICE=${INCIDENCE_SLICE:-1}
if [ "$INCIDENCE_SLICE" -ge 3 ]; then exit 2; fi
INCIDENCE_NEXT=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" \
    --export="ALL,INCIDENCE_SLICE=$((INCIDENCE_SLICE+1))" deltaai/incidence_ablation/run.sbatch)
printf '%s\n' "$INCIDENCE_NEXT" >> deltaai/results/joint_incidence_20260921/jobs.txt

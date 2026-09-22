#!/bin/bash
#SBATCH --job-name=mom-top4
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-5%3
#SBATCH --time=04:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/results/joint_mom_20260921/logs/train-%A_%a.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention
export PYTHONSAFEPATH=1 FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/deltaai/mom_ablation:/u/archerdw/triton371:$PWD/deltaai/results/joint_incidence_20260921/source/deltaai:$PWD/deltaai/results/joint_incidence_20260921/source/smat:$PWD/deltaai/results/joint_incidence_20260921/source:/u/archerdw/.local/lib/python3.12/site-packages"
export TRITON_CACHE_DIR="$PWD/deltaai/results/synthetic_memory/triton_cache"
export MOM_INDEX=${MOM_INDEX:-$SLURM_ARRAY_TASK_ID}
python -P -u deltaai/mom_ablation/run.py --index "$MOM_INDEX" --max-minutes 225
flock deltaai/results/joint_mom_20260921/collect.lock python -P -u deltaai/mom_ablation/collect.py
if python - <<'PY'
import json,os
from pathlib import Path
r=Path('deltaai/results/joint_mom_20260921')
t=json.loads((r/'tasks.json').read_text())[int(os.environ['MOM_INDEX'])]
raise SystemExit(0 if Path(t['result']).exists() else 1)
PY
then exit 0; fi
MOM_SLICE=${MOM_SLICE:-1}
if [ "$MOM_SLICE" -ge 3 ]; then exit 2; fi
MOM_NEXT=$(sbatch --parsable --array="$MOM_INDEX" --dependency="afterok:$SLURM_JOB_ID" \
    --export="ALL,MOM_INDEX=$MOM_INDEX,MOM_SLICE=$((MOM_SLICE+1))" deltaai/mom_ablation/run.sbatch)
printf '%s\n' "$MOM_NEXT" >> deltaai/results/joint_mom_20260921/jobs.txt

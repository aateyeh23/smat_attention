#!/bin/bash
#SBATCH --job-name=paper-ll-seed4
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-1%2
#SBATCH --time=02:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/results/paper_seeds_20260920/logs/dedicated-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/$USER-paper-seeds-triton" PYTHONSAFEPATH=1
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    export PAPER_ONLY=joint-loglinear-mamba2-s4
else
    export PAPER_ONLY=joint-loglinear-gdn-s4
fi
python -u deltaai/paper_seeds/campaign.py --minutes 105 --only "$PAPER_ONLY"
if python - <<'PY'
import json,os
from pathlib import Path
root=Path('deltaai/results/paper_seeds_20260920')
task,=[t for t in json.loads((root/'tasks.json').read_text()) if t['name']==os.environ['PAPER_ONLY']]
path=Path(task['result'])
if not path.exists():raise SystemExit(1)
result=json.loads(path.read_text())
assert result['complete'] and result['epochs']==32 and result['steps']==22560
PY
then exit 0; fi
PAPER_DEDICATED_SLICE=${PAPER_DEDICATED_SLICE:-1}
if [ "$PAPER_DEDICATED_SLICE" -ge 100 ]; then exit 2; fi
PAPER_NEXT=$(sbatch --parsable --array="$SLURM_ARRAY_TASK_ID" --dependency="afterok:$SLURM_JOB_ID" \
    --export="ALL,PAPER_DEDICATED_SLICE=$((PAPER_DEDICATED_SLICE+1))" deltaai/paper_seeds/run_dedicated.sbatch)
printf '%s\n' "$PAPER_NEXT" >> deltaai/results/paper_seeds_20260920/jobs.txt

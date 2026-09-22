#!/bin/bash
#SBATCH --job-name=paper-5seeds
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-7%8
#SBATCH --time=02:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/results/paper_seeds_20260920/logs/slurm-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/$USER-paper-seeds-triton"
export PYTHONSAFEPATH=1
export PAPER_SHARD=${PAPER_SHARD:-$SLURM_ARRAY_TASK_ID}
python -u deltaai/paper_seeds/campaign.py --minutes 105 --shard "$PAPER_SHARD"
if python - <<'PY'
import json,os
from pathlib import Path
root=Path('deltaai/results/paper_seeds_20260920')
tasks=json.loads((root/'tasks.json').read_text())
tasks=tasks[int(os.environ['PAPER_SHARD'])::8]
tasks=[t for t in tasks if not t.get('dedicated')]
remaining=[t for t in tasks if not Path(t['result']).exists()]
print(f'{len(tasks)-len(remaining)}/{len(tasks)} complete',flush=True)
raise SystemExit(bool(remaining))
PY
then exit 0; fi
PAPER_SLICE=${PAPER_SLICE:-1}
if [ "$PAPER_SLICE" -ge 100 ]; then exit 2; fi
PAPER_NEXT=$(sbatch --parsable --partition="$SLURM_JOB_PARTITION" --array="$PAPER_SHARD" --dependency="afterok:$SLURM_JOB_ID" \
    --export="ALL,PAPER_SHARD=$PAPER_SHARD,PAPER_SLICE=$((PAPER_SLICE+1))" deltaai/paper_seeds/run.sbatch)
printf '%s\n' "$PAPER_NEXT" >> deltaai/results/paper_seeds_20260920/jobs.txt

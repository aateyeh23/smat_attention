#!/bin/bash
#SBATCH --job-name=paper-fast-ll
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-9%10
#SBATCH --time=02:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/results/paper_fast_20260920/logs/slurm-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONSAFEPATH=1
export PAPER_FAST_SHARD=${PAPER_FAST_SHARD:-$SLURM_ARRAY_TASK_ID}
python -u deltaai/paper_seeds/fast_campaign.py --minutes 105 --shards 10 --shard "$PAPER_FAST_SHARD"
if python - <<'PY'
import json,os
from pathlib import Path
root=Path('deltaai/results/paper_fast_20260920')
task=json.loads((root/'tasks.json').read_text())[int(os.environ['PAPER_FAST_SHARD'])]
path=Path(task['result'])
if not path.exists():raise SystemExit(1)
result=json.loads(path.read_text())
assert result['complete'] and result['epochs']==32 and result['steps']==22560
PY
then exit 0; fi
PAPER_FAST_SLICE=${PAPER_FAST_SLICE:-1}
if [ "$PAPER_FAST_SLICE" -ge 20 ]; then exit 2; fi
PAPER_NEXT=$(sbatch --parsable --array="$PAPER_FAST_SHARD" --dependency="afterok:$SLURM_JOB_ID" \
    --export="ALL,PAPER_FAST_SHARD=$PAPER_FAST_SHARD,PAPER_FAST_SLICE=$((PAPER_FAST_SLICE+1))" deltaai/paper_seeds/run_fast.sbatch)
printf '%s\n' "$PAPER_NEXT" >> deltaai/results/paper_fast_20260920/jobs.txt

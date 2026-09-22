#!/bin/bash
#SBATCH --job-name=incidence-shard
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-3%4
#SBATCH --time=04:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/results/joint_incidence_20260921/logs/shard-%A_%a.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention
export PYTHONSAFEPATH=1
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/deltaai/incidence_ablation:/u/archerdw/triton371:$PWD/deltaai/results/joint_incidence_20260921/source/deltaai:$PWD/deltaai/results/joint_incidence_20260921/source/smat:$PWD/deltaai/results/joint_incidence_20260921/source:/u/archerdw/.local/lib/python3.12/site-packages"
export TRITON_CACHE_DIR="$PWD/deltaai/results/synthetic_memory/triton_cache"
export INCIDENCE_SHARD=${INCIDENCE_SHARD:-$SLURM_ARRAY_TASK_ID}
python - <<'PY'
import hashlib,json
from pathlib import Path
r=Path('deltaai/results/joint_incidence_20260921')
for p,h in json.loads((r/'scheduler_sha256.json').read_text()).items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
PY
python -P -u deltaai/results/joint_incidence_20260921/source/deltaai/joint_recall_campaign.py \
    --manifest "deltaai/results/joint_incidence_20260921/shards/$INCIDENCE_SHARD/tasks.json" --minutes 225
flock deltaai/results/joint_incidence_20260921/collect.lock python -P -u deltaai/incidence_ablation/collect.py
if python - <<'PY'
import json,os
from pathlib import Path
r=Path('deltaai/results/joint_incidence_20260921/shards')/os.environ['INCIDENCE_SHARD']
tasks=json.loads((r/'tasks.json').read_text())
raise SystemExit(0 if all(Path(t['result']).exists() for t in tasks) else 1)
PY
then exit 0; fi
INCIDENCE_SLICE=${INCIDENCE_SLICE:-1}
if [ "$INCIDENCE_SLICE" -ge 3 ]; then exit 2; fi
INCIDENCE_NEXT=$(sbatch --parsable --array="$INCIDENCE_SHARD" --dependency="afterok:$SLURM_JOB_ID" \
    --export="ALL,INCIDENCE_SHARD=$INCIDENCE_SHARD,INCIDENCE_SLICE=$((INCIDENCE_SLICE+1))" deltaai/incidence_ablation/run_sharded.sbatch)
printf '%s\n' "$INCIDENCE_NEXT" >> "deltaai/results/joint_incidence_20260921/shards/$INCIDENCE_SHARD/jobs.txt"

#!/bin/bash
#SBATCH --job-name=incidence-shard
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-3%4
#SBATCH --time=04:00:00
#SBATCH --output=results/joint_incidence_20260921/logs/shard-%A_%a.out
set -eo pipefail
module load <pytorch-2.11-module> >/dev/null 2>&1
cd .
export PYTHONSAFEPATH=1
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/incidence_ablation:<triton-3.7.1>:$PWD/results/joint_incidence_20260921/source/the GPU cluster:$PWD/results/joint_incidence_20260921/source/smat:$PWD/results/joint_incidence_20260921/source:<site-packages>"
export TRITON_CACHE_DIR="$PWD/results/synthetic_memory/triton_cache"
export INCIDENCE_SHARD=${INCIDENCE_SHARD:-$SLURM_ARRAY_TASK_ID}
python - <<'PY'
import hashlib,json
from pathlib import Path
r=Path('results/joint_incidence_20260921')
for p,h in json.loads((r/'scheduler_sha256.json').read_text()).items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
PY
python -P -u results/joint_incidence_20260921/source/joint_recall_campaign.py \
    --manifest "results/joint_incidence_20260921/shards/$INCIDENCE_SHARD/tasks.json" --minutes 225
flock results/joint_incidence_20260921/collect.lock python -P -u incidence_ablation/collect.py
if python - <<'PY'
import json,os
from pathlib import Path
r=Path('results/joint_incidence_20260921/shards')/os.environ['INCIDENCE_SHARD']
tasks=json.loads((r/'tasks.json').read_text())
raise SystemExit(0 if all(Path(t['result']).exists() for t in tasks) else 1)
PY
then exit 0; fi
INCIDENCE_SLICE=${INCIDENCE_SLICE:-1}
if [ "$INCIDENCE_SLICE" -ge 3 ]; then exit 2; fi
INCIDENCE_NEXT=$(sbatch --parsable --array="$INCIDENCE_SHARD" --dependency="afterok:$SLURM_JOB_ID" \
    --export="ALL,INCIDENCE_SHARD=$INCIDENCE_SHARD,INCIDENCE_SLICE=$((INCIDENCE_SLICE+1))" incidence_ablation/run_sharded.sbatch)
printf '%s\n' "$INCIDENCE_NEXT" >> "results/joint_incidence_20260921/shards/$INCIDENCE_SHARD/jobs.txt"

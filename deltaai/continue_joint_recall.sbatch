#!/bin/bash
#SBATCH --job-name=joint-recall
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/logs/joint-recall-%j.out
set -eo pipefail
cd /u/archerdw/smat_attention
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
JR_SLICE=${JR_SLICE:-1}
check_complete() {
    python - <<'PY'
import json
from pathlib import Path
root=Path('deltaai/results/joint_recall')
names=json.loads((root/'planned_runs.json').read_text())
assert len(names)==24
missing=[]
for name in names:
    path=root/name/'result.json'
    if not path.exists():missing.append(name);continue
    r=json.loads(path.read_text())
    assert r['complete'] and r['epochs']==32 and r['steps']==22624
print(f'{24-len(missing)}/24 completed')
raise SystemExit(10 if missing else 0)
PY
}
if check_complete; then
    python deltaai/audit_joint_recall_results.py
    python deltaai/collect_joint_recall.py
    exit 0
else JR_CHECK=$?; fi
if [ "$JR_CHECK" -ne 10 ]; then exit "$JR_CHECK"; fi
bash deltaai/run_joint_recall.sbatch --max-minutes 110
python deltaai/collect_joint_recall.py
python deltaai/audit_joint_recall_results.py
if check_complete; then exit 0; else JR_CHECK=$?; fi
if [ "$JR_CHECK" -ne 10 ]; then exit "$JR_CHECK"; fi
if [ "$JR_SLICE" -ge 12 ]; then
    echo 'Campaign still incomplete after 12 continuation slices; manual review required.'
    exit 2
fi
JR_NEXT=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" \
    --export="ALL,JR_SLICE=$((JR_SLICE+1))" deltaai/continue_joint_recall.sbatch)
printf '%s\n' "$JR_NEXT" >> deltaai/results/joint_recall/continuation_jobs.txt
echo "Checkpointed incomplete runs; queued continuation $JR_NEXT"

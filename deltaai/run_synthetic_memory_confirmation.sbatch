#!/bin/bash
#SBATCH --job-name=synthetic-confirm
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=2
#SBATCH --time=00:35:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/logs/synthetic-memory-%j.out
set -eo pipefail
cd /u/archerdw/smat_attention
export SYNTHETIC_DIMS=1,2,4,3
COMMON=(--steps 500 --families mamba2,gdn_current --seeds 123,456,789
        --min-length 64 --eval-every 100 --eval-examples 1024 --test-examples 4096)
# Run only after inspecting/snapshotting the complete one-seed sweeps.
# The existing seed123 cells are reused (evaluation backfilled when necessary).
SYNTHETIC_LOG_TAG=screen-state bash deltaai/run_synthetic_memory_parallel.sbatch \
    "${COMMON[@]}" --task state --stage screen500 --width 32 --loads 4 --max-minutes 8
SYNTHETIC_LOG_TAG=main-state bash deltaai/run_synthetic_memory_parallel.sbatch \
    "${COMMON[@]}" --task state --stage main500 --width 64 --loads 4,16 --max-minutes 11
SYNTHETIC_LOG_TAG=main-variable bash deltaai/run_synthetic_memory_parallel.sbatch \
    "${COMMON[@]}" --task variable --stage main500 --width 64 --loads 2,8 --max-minutes 11
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
python - <<'PY'
import json
from pathlib import Path
root = Path('deltaai/results/synthetic_memory')
for stage, task, width, loads in [('screen500','state',32,[4]),
                                ('main500','state',64,[4,16]),
                                ('main500','variable',64,[2,8])]:
    for n in loads:
        for seed in [123,456,789]:
            for family in ['mamba2','gdn_current']:
                for d in [1,2,3,4]:
                    p = root/stage/task/f'w{width}-n{n}-{family}-d{d}-s{seed}'/'result.json'
                    r = json.loads(p.read_text())
                    assert r['complete'] and r['steps']==500 and 'binding_gap_pp' in r, p
print('All 120 selected confirmation cells complete (40 reused, 80 new).')
PY

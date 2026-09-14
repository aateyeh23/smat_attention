#!/bin/bash
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention/deltaai
export PYTHONPATH="/u/archerdw/triton371:$PWD:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-}"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_CACHE_DIR="$PWD/results/mqar_gdn_reset/check_cache"
python -u diagnose_mqar_gdn.py > results/mqar_gdn_reset/w16w32_s123/diagnosis.log 2>&1

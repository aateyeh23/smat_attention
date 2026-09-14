#!/bin/bash
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/an author/smat_attention/the GPU cluster
export PYTHONPATH="/u/an author/triton371:$PWD:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology" FLA_TILELANG=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled TRITON_F32_DEFAULT=tf32x3
export TRITON_CACHE_DIR="$PWD/results/mqar_gdn_reset/check_cache"
python -u test_gdn_heads.py > results/sc_gdn_smat/check/alignment.log 2>&1
python -u test_delta_shards.py > results/sc_gdn_smat/check/shards.log 2>&1
touch results/sc_gdn_smat/check/validated

#!/bin/bash
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0
cd /u/archerdw/smat_attention/deltaai
export PYTHONPATH=/u/archerdw/triton371:$PWD:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology
export FLA_TILELANG=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TRITON_F32_DEFAULT=tf32x3
export TRITON_CACHE_DIR=$PWD/results/mqar_gdn_reset/check_cache
python test_causal_keys_widths.py

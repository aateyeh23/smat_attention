#!/bin/bash
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0
cd /u/an author/smat_attention/the GPU cluster
export PYTHONPATH=/u/an author/triton371:$PWD:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology
export FLA_TILELANG=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TRITON_F32_DEFAULT=tf32x3
export TRITON_CACHE_DIR=$PWD/results/mqar_gdn_reset/check_cache
python analysis/verify_soft_routing.py

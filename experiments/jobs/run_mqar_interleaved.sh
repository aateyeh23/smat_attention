#!/bin/bash
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/an author/smat_attention/the GPU cluster
export PYTHONPATH="/u/an author/triton371:$PWD:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-}"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export ZOO_EPOCHS=32 ZOO_DS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MQAR_RUN_DIR="$PWD/results/mqar_gdn_interleaved/quoted_heads_s123"
export TRITON_CACHE_DIR="$PWD/results/mqar_gdn_reset/check_cache"
mkdir -p "$MQAR_RUN_DIR"
cp zoo_mqar_interleave.py experiments/zoology/gdn_interleaved.py experiments/zoology/gdn_reset.py zoo_smat_gdn.py zoo_mqar_resume.py run_mqar_interleaved.sh "$MQAR_RUN_DIR/"
pids=()
for width in ${MQAR_INTERLEAVE_WIDTHS:-16 32}; do
    env ZOO_DM="$width" python -u -m zoology.launch experiments/zoology/gdn_interleaved.py >> "$MQAR_RUN_DIR/w${width}-d3.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"

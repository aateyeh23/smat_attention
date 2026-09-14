#!/bin/bash
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention/deltaai
export PYTHONPATH="/u/archerdw/triton371:$PWD:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-}"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export ZOO_DM=16 ZOO_EPOCHS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MQAR_RUN_DIR="$PWD/results/mqar_mamba_reset/w16_independent_s123"
export TRITON_CACHE_DIR="$PWD/results/mqar_gdn_reset/check_cache"
mkdir -p "$MQAR_RUN_DIR"
cp zoo_mamba_reset_configs.py zoo_mqar_resume.py zoo_smat_mixer.py run_mqar_mamba_w16.sh "$MQAR_RUN_DIR/"
python -u test_mqar_mamba_reset.py >> "$MQAR_RUN_DIR/test.log" 2>&1
pids=()
for d in ${M2_DS:-1 2 3 4}; do
    env ZOO_DS="$d" python -u -m zoology.launch zoo_mamba_reset_configs.py >> "$MQAR_RUN_DIR/w16-d${d}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"

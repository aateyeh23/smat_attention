#!/bin/bash
#SBATCH --job-name=synthetic-memory-par
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/logs/synthetic-memory-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention/deltaai
export PYTHONPATH="/u/archerdw/triton371:$PWD:$PWD/../smat:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/results/synthetic_memory/triton_cache"
if [ -n "${SYNTHETIC_REQUIRE_COMPLETE:-}" ] && [ ! -f "$SYNTHETIC_REQUIRE_COMPLETE" ]; then
    echo "Preceding pilot is incomplete: $SYNTHETIC_REQUIRE_COMPLETE"
    exit 2
fi
if [ "${SYNTHETIC_REPORT_BEFORE:-0}" = "1" ]; then
    python collect_synthetic_memory.py
fi
IFS=',' read -ra SMEM_GPUS <<< "$CUDA_VISIBLE_DEVICES"
IFS=',' read -ra SMEM_DIMS <<< "${SYNTHETIC_DIMS:-1,2,3,4}"
# Distribute d groups over whatever GPU count Slurm allocated (1, 2, or 4).
SMEM_GROUPS=()
for i in "${!SMEM_DIMS[@]}"; do
    SMEM_G=$((i % ${#SMEM_GPUS[@]}))
    if [ -n "${SMEM_GROUPS[$SMEM_G]:-}" ]; then
        SMEM_GROUPS[$SMEM_G]+="," 
    fi
    SMEM_GROUPS[$SMEM_G]+="${SMEM_DIMS[$i]}"
done
SMEM_PIDS=()
for i in "${!SMEM_GROUPS[@]}"; do
    SMEM_D="${SMEM_GROUPS[$i]}"
    SMEM_TAG="${SMEM_D//,/-}"
    CUDA_VISIBLE_DEVICES="${SMEM_GPUS[$i]}" python -u synthetic_memory_campaign.py "$@" \
        --ds "$SMEM_D" --completion-tag="-ds$SMEM_TAG" \
        > "logs/synthetic-memory-${SLURM_JOB_ID}${SYNTHETIC_LOG_TAG:+-$SYNTHETIC_LOG_TAG}-ds${SMEM_TAG}.out" 2>&1 &
    SMEM_PIDS+=("$!")
    echo "GPU ${SMEM_GPUS[$i]}: d=$SMEM_D, pid=$!"
done
SMEM_RC=0
for SMEM_PID in "${SMEM_PIDS[@]}"; do
    wait "$SMEM_PID" || SMEM_RC=1
done
exit "$SMEM_RC"

#!/bin/bash
#SBATCH --job-name=joint-recall
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/logs/joint-recall-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention/deltaai
export PYTHONPATH="/u/archerdw/triton371:$PWD:$PWD/../smat:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/results/synthetic_memory/triton_cache"
JR_ROOT="$PWD/results/joint_recall"
JR_CODE="$PWD"
if [ -d "$JR_ROOT/source/deltaai" ]; then JR_CODE="$JR_ROOT/source/deltaai"; fi
python -u "$JR_CODE/joint_recall.py" --root "$JR_ROOT" "$@" --prepare-only
IFS=',' read -ra JR_GPUS <<< "$CUDA_VISIBLE_DEVICES"
JR_WORKERS=${#JR_GPUS[@]}
if [ "$JR_WORKERS" -gt 1 ]; then
    JR_PIDS=()
    for ((i=0; i<JR_WORKERS; i++)); do
        CUDA_VISIBLE_DEVICES="${JR_GPUS[$i]}" python -u "$JR_CODE/joint_recall.py" --root "$JR_ROOT" "$@" --worker "$i" --workers "$JR_WORKERS" \
            > "logs/joint-recall-${SLURM_JOB_ID}-worker${i}.out" 2>&1 &
        JR_PIDS+=("$!")
    done
    JR_RC=0
    for JR_PID in "${JR_PIDS[@]}"; do wait "$JR_PID" || JR_RC=1; done
    exit "$JR_RC"
fi
python -u "$JR_CODE/joint_recall.py" --root "$JR_ROOT" "$@"

#!/bin/bash
#SBATCH --job-name=mqar-w16-read4
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/u/archerdw/smat_attention/deltaai/logs/mqar-four-reads-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
: "${MQAR_FOUR_READS_DIR:?Set the isolated result directory}"
FOUR_READS_CODE="$MQAR_FOUR_READS_DIR/source/deltaai"
cd "$FOUR_READS_CODE"
export PYTHONPATH="/u/archerdw/triton371:$FOUR_READS_CODE:$MQAR_FOUR_READS_DIR/source/smat:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/u/archerdw/smat_attention/deltaai/results/mqar_gdn_reset/check_cache
unset MQAR_RUN_DIR ZOO_SMOKE
if [ -f "$MQAR_FOUR_READS_DIR/COMPLETE" ]; then exit 0; fi
FOUR_READS_NEXT=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" "$FOUR_READS_CODE/run_mqar_four_reads.sbatch")
echo "$SLURM_JOB_ID $FOUR_READS_NEXT" >> "$MQAR_FOUR_READS_DIR/job_chain.txt"
set +e
python -u run_mqar_four_reads.py
FOUR_READS_RC=$?
set -e
if [ -f "$MQAR_FOUR_READS_DIR/COMPLETE" ] || { [ "$FOUR_READS_RC" -ne 0 ] && [ "$FOUR_READS_RC" -lt 128 ]; }; then
    scancel "$FOUR_READS_NEXT"
fi
exit "$FOUR_READS_RC"

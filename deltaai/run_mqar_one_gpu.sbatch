#!/bin/bash
#SBATCH --job-name=mqar-one-gpu
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mqar-one-gpu-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH="/u/archerdw/triton371:$PWD:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-}"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/results/mqar_gdn_reset/check_cache"
export MQAR_CONCURRENCY=4 MQAR_SLICE_MINUTES=100
unset MQAR_RUN_DIR
run_dir="$PWD/results/mqar_one_gpu/${SLURM_JOB_ID}"
mkdir -p "$run_dir"
cp mqar_one_gpu.py run_mqar_one_gpu.sbatch zoo_mqar_resume.py zoo_mqar_interleave.py zoo_smat_gdn.py zoo_smat_mixer.py zoo_gdn_reset_configs.py zoo_gdn_interleaved_configs.py zoo_mamba_reset_configs.py "$run_dir/"
# Queue the next slice in advance: time-limit interruption still resumes from checkpoints.
next_job=$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" run_mqar_one_gpu.sbatch)
echo "CONTINUATION JOB $next_job"
cleanup_failure() {
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -lt 128 ]; then
        scancel "$next_job" || true
        echo "MQAR error: stopped continuation for diagnosis"
    fi
}
trap cleanup_failure EXIT
for width in 16 32 64; do
    env ZOO_DM="$width" python -u test_gdn_reset.py > "$run_dir/test-gdn-w${width}.log" 2>&1
done
python -u test_mqar_mamba_reset.py > "$run_dir/test-mamba.log" 2>&1
status=0
python -u mqar_one_gpu.py || status=$?
if [ "$status" -eq 0 ]; then
    scancel "$next_job"
    echo "ALL MQAR RUNS COMPLETE"
    scontrol update JobId=3143305 Partition=ghx4-interactive Nice=0 || true
    scontrol release 3143305 || true
elif [ "$status" -eq 2 ]; then
    echo "Slice checkpointed; continuing in job $next_job"
else
    exit "$status"
fi

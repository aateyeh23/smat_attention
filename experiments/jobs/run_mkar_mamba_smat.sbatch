#!/bin/bash
#SBATCH --job-name=smat-mkar-mamba
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x-%j.out

# Learned Mamba-2 + SMat (d=4, one hashed write, four learned reads) on
# multi-key subset recall: one seed per task, a separate model for each of
# k = 1, 2, 3, 12000 steps, FP32 (results/mkar_mamba_smat_20260923).
#
#   experiments/submit.sh experiments/jobs/run_mkar_mamba_smat.sbatch validate
#   experiments/submit.sh experiments/jobs/run_mkar_mamba_smat.sbatch train 0   # tasks 0..3
#
# MKAR_CAMPAIGN is confirm (seeds 456, 789, 2026, 2027; the default) or
# pilot_s123 (seed 123, task 0).  MKAR_ROOT is where runs are written; it starts
# from the recorded protocol, task list and source manifest, so the validation
# and training check the same recipe the recorded runs used.  Each job trains
# for at most 50 minutes and resubmits itself from its checkpoint.
set -eo pipefail
source experiments/prelude.sh

export MKAR_CAMPAIGN="${MKAR_CAMPAIGN:-confirm}"
export MKAR_ROOT="${MKAR_ROOT:-${SMAT_WORK:-$PWD/data}/mkar_mamba_smat/$MKAR_CAMPAIGN}"
record=results/mkar_mamba_smat_20260923/$MKAR_CAMPAIGN
mkdir -p "$MKAR_ROOT"
for f in protocol.json tasks.json source_sha256.json; do
    [ -f "$MKAR_ROOT/$f" ] || cp "$record/$f" "$MKAR_ROOT/"
done

export PYTHONPATH="$PWD/tasks/mkar_mamba_smat:$PYTHONPATH"
export PYTHONSAFEPATH=1 FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_F32_DEFAULT=tf32x3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/mkar-triton-${SLURM_JOB_ID:-$$}"
mkdir -p "$TRITON_CACHE_DIR"

case "${1:-train}" in
  validate)
    python -P -u tasks/mkar_mamba_smat/validate.py ;;
  train)
    index="${2:?task index}"
    trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1 TERM
    python -P -u tasks/mkar_mamba_smat/train.py --index "$index" --max-minutes 50 &
    pid=$!
    set +e
    wait "$pid"; rc=$?
    if kill -0 "$pid" 2>/dev/null; then wait "$pid"; rc=$?; fi
    set -e
    if [ "$rc" -eq 75 ]; then      # out of time: continue from the checkpoint
        exec experiments/submit.sh experiments/jobs/run_mkar_mamba_smat.sbatch train "$index"
    fi
    flock "$MKAR_ROOT/collect.lock" python -P -u tasks/mkar_mamba_smat/collect.py
    exit "$rc" ;;
  *)
    echo "usage: $0 validate | train <index>" >&2; exit 2 ;;
esac

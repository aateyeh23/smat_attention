#!/bin/bash
#SBATCH --job-name=smat-winsched
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/winsched-%j.out
# Boundary-schedule study (tasks/window_schedules.py).  Runs every command line
# given in the file $1 concurrently on the one allocated GPU, then waits.
#   sbatch experiments/jobs/run_window_schedules.sbatch experiments/jobs/winsched_<group>.txt
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
[ -f experiments/site.conf ] && . experiments/site.conf
[ -n "${SMAT_ENV:-}" ] && eval "$SMAT_ENV"
echo "start $(date)"
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" || exit 1
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "launch: $line"
  bash -c "python -u tasks/window_schedules.py $line" &
done < "$1"
wait
echo "done $(date)"

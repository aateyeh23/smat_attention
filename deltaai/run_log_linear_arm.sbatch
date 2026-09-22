#!/bin/bash
#SBATCH --job-name=ll-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# usage: sbatch --job-name=ll-gdn-w16 run_log_linear_arm.sbatch gdn 16
set -uo pipefail
source _prelude_ll_scf.sh
python -u run_log_linear_arm.py "$1" "$2" --max-minutes "${MQAR_MAX_MINUTES:-690}"
rc=$?
echo "ARM EXIT $rc  $(date)"
exit $rc

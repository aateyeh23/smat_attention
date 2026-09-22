#!/bin/bash
#SBATCH --job-name=ll-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
source _prelude_ll_scf.sh
python -u run_log_linear_validate.py

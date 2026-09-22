#!/bin/bash
#SBATCH --job-name=mqar-backlog
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mqar-backlog-%j.out
set -eo pipefail
srun --kill-on-bad-exit=0 bash /u/archerdw/smat_attention/deltaai/results/mqar_gdn_four_reads/w16_h1_hd16_state16_s123/source/deltaai/mqar_dispatch_env.sh bulk --manifest "$1" --minutes 100

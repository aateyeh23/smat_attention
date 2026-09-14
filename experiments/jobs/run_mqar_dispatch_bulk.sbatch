#!/bin/bash
#SBATCH --job-name=mqar-backlog
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
srun --kill-on-bad-exit=0 bash /u/an author/smat_attention/the GPU cluster/mqar_dispatch_env.sh bulk --manifest "$1" --minutes 100

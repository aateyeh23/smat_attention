#!/bin/bash
#SBATCH --job-name=gdn-mqar-backfill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --chdir=/u/an author/smat_attention/the GPU cluster
#SBATCH --output=logs/%x-%j.out
# Shared dispatcher claims prevent duplicates with the interactive/multi-node workers.
exec bash /u/an author/smat_attention/the GPU cluster/mqar_dispatch_env.sh bulk --manifest "$1" --minutes 15

#!/bin/bash
#SBATCH --job-name=gdn-mqar-backfill
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --chdir=/u/archerdw/smat_attention/deltaai
#SBATCH --output=/u/archerdw/smat_attention/deltaai/logs/gdn-mqar-backfill-%j.out
# Shared dispatcher claims prevent duplicates with the interactive/multi-node workers.
exec bash /u/archerdw/smat_attention/deltaai/results/mqar_gdn_four_reads/w16_h1_hd16_state16_s123/source/deltaai/mqar_dispatch_env.sh bulk --manifest "$1" --minutes 15

#!/bin/bash
#SBATCH --job-name=mqar-sequential
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mqar-sequential-%j.out
exec bash /u/archerdw/smat_attention/deltaai/results/mqar_gdn_four_reads/w16_h1_hd16_state16_s123/source/deltaai/mqar_dispatch_env.sh single --minutes 100

#!/bin/bash
#SBATCH --job-name=mqar-sequential
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
exec bash /u/an author/smat_attention/the GPU cluster/mqar_dispatch_env.sh single --minutes 100

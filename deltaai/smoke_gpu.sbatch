#!/bin/bash
#SBATCH --job-name=smat-smoke
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:35:00
#SBATCH --output=logs/smoke-%j.out

# DeltaAI (GH200) port of ../smat/smoke_gpu.sbatch.  Same content: correctness
# with the Triton kernels, then a tiny bf16 sweep.  Outputs stay in this folder.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"

echo "==== correctness incl. triton kernels ===="
OMP_NUM_THREADS=4 python -u "$SMAT/test_smat.py" --skip-vc

echo "==== tiny gpu sweep ===="
python -u "$SMAT/bench_smat.py" --device cuda --dtype bf16 \
    --T 4096 8192 16384 32768 --d 1 2 3 4 \
    --nb 8 --r 64 --d-qk 64 --d-v 64 --chunk 128 \
    --repeat 8 --warmup 3 --out results/smoke_bf16.csv
echo "done: $(date)"

#!/bin/bash
#SBATCH --job-name=mqar-gdn-reset
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mqar-gdn-reset-%j.out
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH="/u/archerdw/triton371:$PWD:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-}"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export ZOO_DM=${ZOO_DM:-32} ZOO_EPOCHS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_DIR="$PWD/results/mqar_gdn_reset/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$RUN_DIR/triton_cache"
mkdir -p "$RUN_DIR"
cp zoo_smat_gdn.py zoo_gdn_reset_configs.py test_gdn_reset.py run_mqar_gdn_reset.sbatch "$RUN_DIR/"
python -u test_gdn_reset.py > "$RUN_DIR/tests.log" 2>&1
cat "$RUN_DIR/tests.log"
pids=()
for d in 1 2 3 4; do
    env ZOO_DS="$d" python -u -m zoology.launch zoo_gdn_reset_configs.py > "$RUN_DIR/d${d}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "FINISHED status=$status results=$RUN_DIR"
exit "$status"

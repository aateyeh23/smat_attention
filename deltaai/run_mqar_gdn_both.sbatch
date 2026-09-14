#!/bin/bash
#SBATCH --job-name=mqar-gdn-plus-m2w16
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/mqar-gdn-both-%j.out
set -eo pipefail
continuation_queued=0
restore_copying() {
    if [ "$continuation_queued" -eq 0 ]; then
        scontrol update JobId=3143305 Partition=ghx4-interactive Nice=0 || true
        scontrol release 3143305 || true
    fi
}
trap restore_copying EXIT
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH="/u/archerdw/triton371:$PWD:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-}"
export FLA_TILELANG=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export ZOO_EPOCHS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MQAR_RUN_DIR="$PWD/results/mqar_gdn_reset/quoted_heads_s123"
export TRITON_CACHE_DIR="$PWD/results/mqar_gdn_reset/check_cache"
mkdir -p "$MQAR_RUN_DIR"
cp zoo_smat_gdn.py zoo_gdn_reset_configs.py zoo_mqar_resume.py run_mqar_gdn_both.sbatch "$MQAR_RUN_DIR/"
for width in 16 32 64; do
    env ZOO_DM="$width" python -u test_gdn_reset.py > "$MQAR_RUN_DIR/test-w${width}.log" 2>&1
done
pids=()
gpu=0
for width in 16 32 64; do
    for d in 1 2 3 4; do
        env CUDA_VISIBLE_DEVICES="$gpu" ZOO_DM="$width" ZOO_DS="$d" \
            python -u -m zoology.launch zoo_gdn_reset_configs.py >> "$MQAR_RUN_DIR/w${width}-d${d}.log" 2>&1 &
        pids+=("$!")
    done
    gpu=$((gpu + 1))
done
env CUDA_VISIBLE_DEVICES=3 M2_DS="1 2 3 4" bash run_mqar_mamba_w16.sh &
pids+=("$!")
env CUDA_VISIBLE_DEVICES=0 MQAR_INTERLEAVE_WIDTHS=16 bash run_mqar_interleaved.sh &
pids+=("$!")
env CUDA_VISIBLE_DEVICES=1 MQAR_INTERLEAVE_WIDTHS=32 bash run_mqar_interleaved.sh &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ "$status" -eq 0 ]; then
    if python -c 'import json, os; from pathlib import Path; p=Path(os.environ["MQAR_RUN_DIR"]); paths=[p/f"w{w}-d{d}.json" for w in (16,32,64) for d in (1,2,3,4)]+[Path("results/mqar_mamba_reset/w16_hd16_ds16_independent_s123")/f"w16-d{d}.json" for d in (1,2,3,4)]+[Path("results/mqar_gdn_interleaved/quoted_heads_s123")/f"w{w}-d3.json" for w in (16,32)]; raise SystemExit(0 if all(f.exists() and json.loads(f.read_text())["complete"] for f in paths) else 1)'; then
        echo "ALL EIGHTEEN MQAR RUNS COMPLETE"
    elif [ "${MQAR_SLICE:-1}" -lt 3 ]; then
        sbatch --export="ALL,MQAR_SLICE=$((${MQAR_SLICE:-1} + 1))" run_mqar_gdn_both.sbatch
        continuation_queued=1
    else
        echo "MQAR incomplete after three slices; checkpoints saved"; status=2
    fi
fi
exit "$status"

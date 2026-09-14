#!/bin/bash
#SBATCH --job-name=sc-write-correction-d2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Reset d=2: shared-initialized multiplicative G write correction, seeds 0/1/2.
# All other settings match the earlier reset + read gate + SSM hash run.
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH="$PWD:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
set -u
SC_RUN_ID=${SC_RUN_ID:-${SLURM_JOB_ID}}
SC_SLICE=${SC_SLICE:-1}
SC_D=2
case "$SC_D" in 2|3|4) ;; *) echo "SC_D must be 2, 3, or 4"; exit 1 ;; esac
RUN_DIR="$PWD/results/sc_write_correction/${SC_RUN_ID}"
export TRITON_CACHE_DIR="$PWD/results/sc_write_gate/cache-3143304"
mkdir -p "$RUN_DIR"
if [ ! -f "$RUN_DIR/source.sha256" ]; then
    cp tasks/copying.py zoo_smat_mixer.py content_addr.py test_residual_g_write.py run_sc_write_correction.sbatch "$RUN_DIR/"
    sha256sum tasks/copying.py zoo_smat_mixer.py content_addr.py smat_pool_ops.py test_residual_g_write.py > "$RUN_DIR/source.sha256"
else
    sha256sum -c "$RUN_DIR/source.sha256"
fi
if [ ! -f "$RUN_DIR/validated" ]; then
    python -u test_residual_g_write.py > "$RUN_DIR/tests.log" 2>&1
    cat "$RUN_DIR/tests.log"
fi

COMMON=(--arm smat --d "$SC_D" --reset 1 --seq_len 4096 --n_copy 16 --n_vocab 16
        --d_model 64 --n_layers 2 --headdim 64 --d_state 16 --batch 32
        --steps 10000 --warmup 500 --lr 1e-3 --wd 0.1 --hash_conv_width 4
        --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01
        --balance_gated 1 --eval_every 250 --ckpt_every 250 --log_every 50
        --max_minutes 100 --g_write_init 0.1)

# A full-length smoke run checks the trainer, diagnostics, optimizer and resume.
if [ ! -f "$RUN_DIR/validated" ]; then
    python -u tasks/copying.py "${COMMON[@]}" --g_write_mode residual --steps 2 \
        --batch 2 --eval_every 2 --ckpt_every 2 --log_every 1 --ckpt "$RUN_DIR/smoke.pt" > "$RUN_DIR/smoke.log" 2>&1
    python -u tasks/copying.py "${COMMON[@]}" --g_write_mode residual --steps 3 \
        --batch 2 --eval_every 3 --ckpt_every 3 --log_every 1 --ckpt "$RUN_DIR/smoke.pt" >> "$RUN_DIR/smoke.log" 2>&1
    grep -E 'optimizer covers|resumed|FINAL|G_WRITE|SOURCE_HALF' "$RUN_DIR/smoke.log"
    touch "$RUN_DIR/validated"
fi
if [ "${VALIDATE_ONLY:-0}" = 1 ]; then
    exit 0
fi

pids=()
for seed in 0 1 2; do
    for mode in residual; do
        name="reset-d${SC_D}-${mode}-s${seed}"
        if grep -qE '^FINAL step 10000 |^SOLVED at step ' "$RUN_DIR/$name.log" 2>/dev/null; then
            continue
        fi
        python -u tasks/copying.py "${COMMON[@]}" --seed "$seed" --g_write_mode "$mode" \
            --ckpt "$RUN_DIR/$name.pt" >> "$RUN_DIR/$name.log" 2>&1 &
        pids+=("$!")
    done
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
grep -H -E 'FINAL|SOLVED|TIME LIMIT|Traceback' "$RUN_DIR"/reset-*.log || true
if [ "$status" -eq 0 ]; then
    complete=0
    for log in "$RUN_DIR"/reset-*.log; do
        if grep -qE '^FINAL step 10000 |^SOLVED at step ' "$log"; then
            complete=$((complete + 1))
        fi
    done
    if [ "$complete" -eq 3 ]; then
        echo "ALL THREE RUNS COMPLETE: $RUN_DIR"
    elif [ "$SC_SLICE" -lt 4 ]; then
        # Submit only after clean checkpoint exits; at most four two-hour allocations.
        sbatch --job-name="sc-write-correction-d${SC_D}" --export="ALL,SC_D=$SC_D,SC_RUN_ID=$SC_RUN_ID,SC_SLICE=$((SC_SLICE + 1)),VALIDATE_ONLY=0" run_sc_write_correction.sbatch
    else
        echo "INCOMPLETE after four slices; resumable checkpoints: $RUN_DIR"
        status=2
    fi
fi
exit "$status"

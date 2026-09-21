#!/bin/bash
#SBATCH --job-name=sc-write-gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/sc-write-gate-%j.out
# Reset d=3: shared dt writes vs independent sigmoid writes, seeds 0/1/2.
# All other settings match the earlier reset + read gate + SSM hash run.
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH="$PWD:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TRITON_CACHE_DIR="$PWD/results/sc_write_gate/cache-${SLURM_JOB_ID}"
set -u
RUN_DIR="$PWD/results/sc_write_gate/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
cp lm/sc_train.py zoo_smat_mixer.py content_addr.py test_g_write_gate.py run_sc_write_gate.sbatch "$RUN_DIR/"
sha256sum lm/sc_train.py zoo_smat_mixer.py content_addr.py smat_pool_ops.py > "$RUN_DIR/source.sha256"
python -u test_g_write_gate.py > "$RUN_DIR/tests.log" 2>&1
cat "$RUN_DIR/tests.log"

COMMON=(--arm smat --d 3 --reset 1 --seq_len 4096 --n_copy 16 --n_vocab 16
        --d_model 64 --n_layers 2 --headdim 64 --d_state 16 --batch 32
        --steps 10000 --warmup 500 --lr 1e-3 --wd 0.1 --hash_conv_width 4
        --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01
        --balance_gated 1 --eval_every 250 --ckpt_every 250 --log_every 50
        --max_minutes 460 --g_write_init 0.1)

# A full-length smoke run checks the trainer, diagnostics, optimizer and resume.
python -u lm/sc_train.py "${COMMON[@]}" --g_write_mode independent --steps 2 \
    --batch 2 --eval_every 2 --ckpt_every 2 --log_every 1 --ckpt "$RUN_DIR/smoke.pt" > "$RUN_DIR/smoke.log" 2>&1
python -u lm/sc_train.py "${COMMON[@]}" --g_write_mode independent --steps 3 \
    --batch 2 --eval_every 3 --ckpt_every 3 --log_every 1 --ckpt "$RUN_DIR/smoke.pt" >> "$RUN_DIR/smoke.log" 2>&1
grep -E 'optimizer covers|resumed|FINAL|G_WRITE|SOURCE_HALF' "$RUN_DIR/smoke.log"
if [ "${VALIDATE_ONLY:-0}" = 1 ]; then
    exit 0
fi

pids=()
for seed in 0 1 2; do
    for mode in shared independent; do
        name="reset-d3-${mode}-s${seed}"
        python -u lm/sc_train.py "${COMMON[@]}" --seed "$seed" --g_write_mode "$mode" \
            --ckpt "$RUN_DIR/$name.pt" > "$RUN_DIR/$name.log" 2>&1 &
        pids+=("$!")
    done
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
grep -H -E 'FINAL|SOLVED|TIME LIMIT|Traceback' "$RUN_DIR"/reset-*.log || true
exit "$status"

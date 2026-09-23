#!/bin/bash
#SBATCH -J smat-e2e
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 06:00:00
#SBATCH -o logs/smat-e2e-%j.out
# End-to-end cost of the PG-19 models: one mode per job, every arm and length
# on the same GPU, one process per measurement (see bench_e2e.py for why).
#
#   experiments/submit.sh tasks/pg19/bench_e2e.sbatch train   [rows.jsonl]
#   experiments/submit.sh tasks/pg19/bench_e2e.sbatch prefill [rows.jsonl]
#   experiments/submit.sh tasks/pg19/bench_e2e.sbatch decode  [rows.jsonl]
#   experiments/submit.sh tasks/pg19/bench_e2e.sbatch smoke   [rows.jsonl]
#
# SMAT_BENCH_ENV (site.conf) activates an environment with torch, fla,
# causal_conv1d and mamba_ssm matching the Modal image of the campaign.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f experiments/site.conf ] && . experiments/site.conf
[ -n "${SMAT_BENCH_ENV:-}" ] && eval "$SMAT_BENCH_ENV"
MODE=${1:?train, prefill, decode or smoke}
OUT=${2:-results/pg19_e2e_efficiency/rows.jsonl}

# bench_e2e.py puts src/smat_lm, third_party (the frozen zoology the campaign
# ran) and third_party/log_linear on the path itself.
export SMAT_WRITE_HASH_BACKEND=triton
# Code, venv and results sit on NFS.  Keep compiler caches and bytecode off it:
# a stalled NFS request at interpreter start hung the first submission.
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/$USER-smat-e2e-triton}
mkdir -p "$TRITON_CACHE_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

ARMS="gdn gdn-smat-d2 gdn-smat-d3 gdn-smat-d3-nog mamba2 mamba2-clean mamba2-fused mamba2-smat-d2 mamba2-smat-d3 softmax loglinear-mamba2"
DECODE_ARMS="gdn gdn-smat-d2 gdn-smat-d3 mamba2 mamba2-smat-d2 mamba2-smat-d3 softmax"
# One measurement never takes the job down: a hang or crash costs that row only.
run() { echo "== $*"; timeout 1200 python -u tasks/pg19/bench_e2e.py --out "$OUT" "$@" 2>&1 \
        | grep -E '^ROW|Error|error' | awk '/^ROW /{print; next} {print substr($0, 1, 400)}'; [ "${PIPESTATUS[0]}" = 124 ] && echo "TIMEOUT $*"; }

case "$MODE" in
  smoke)
    for a in $ARMS; do run --arm $a --mode train --length 4096 --warmup 1 --steps 2; done
    for a in $DECODE_ARMS; do run --arm $a --mode decode --length 4096 --decode-steps 8; done ;;
  train)
    for T in 4096 8192 16384 32768; do for a in $ARMS; do run --arm $a --mode train --length $T; done; done
    # the campaign's microbatch, to compare with its recorded tokens/s
    for a in $ARMS; do run --arm $a --mode train --length 16384 --batch 2; done ;;
  prefill)
    for T in 4096 8192 16384 32768; do for a in $ARMS; do run --arm $a --mode prefill --length $T; done; done ;;
  decode)
    for a in $DECODE_ARMS; do
      for b in 1 8; do run --arm $a --mode decode --length 16384 --batch $b; done
      for T in 8192 32768; do run --arm $a --mode decode --length $T; done
    done ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "BENCH $MODE DONE"

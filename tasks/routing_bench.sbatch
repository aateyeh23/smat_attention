#!/bin/bash
#SBATCH -J smat-routing-bench
#SBATCH -c 8
#SBATCH --mem 96G
#SBATCH -t 04:00:00
#SBATCH -o logs/smat-routing-bench-%j.out
# End-to-end cost of the d-Subset Routing models (tasks/routing_bench.py).
#
#   sbatch ... tasks/routing_bench.sbatch smoke|train|prefill|decode|all [rows.jsonl]
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f experiments/site.conf ] && . experiments/site.conf
[ -n "${SMAT_BENCH_ENV:-}" ] && eval "$SMAT_BENCH_ENV"
MODE=${1:?smoke, train, prefill, decode or all}
OUT=${2:-results/routing_bench/$MODE.jsonl}
# bytecode and compiler caches off NFS (a stalled NFS request hung an earlier job)
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/$USER-smat-routing-triton}
mkdir -p "$TRITON_CACHE_DIR" logs
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
run() { echo "== $*"; timeout 3600 python -u tasks/routing_bench.py --out "$OUT" "$@" 2>&1 \
        | grep -E '^ROW|Error|error' | awk '/^ROW /{print; next} {print substr($0, 1, 400)}'; }
case "$MODE" in
  smoke)   for m in train prefill decode; do run --mode $m --T 1024 4096 --warmup 1 --steps 2 --decode-steps 8; done ;;
  train)   run --mode train   --T 1024 4096 16384 65536 ;;
  prefill) run --mode prefill --T 1024 4096 16384 65536 262144 ;;
  decode)  run --mode decode  --T 1024 4096 16384 65536 262144 --nb 1 64 ;;
  all)     # every mode on one GPU, so the three sweeps share hardware
           run --mode train   --T 1024 4096 16384 65536
           run --mode prefill --T 1024 4096 16384 65536 262144
           run --mode decode  --T 1024 4096 16384 65536 262144 --nb 1 64 ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "BENCH $MODE DONE"

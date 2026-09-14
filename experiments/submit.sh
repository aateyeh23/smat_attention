#!/bin/bash
# Submit a job from experiments/jobs with the cluster settings kept out of the
# scripts.  The scripts carry only what the experiment needs; partition,
# account, node list and walltime come from site.conf and are passed on the
# command line, where they override any directive in the file.
#
#   experiments/submit.sh jobs/run_mkar_fair2.sbatch base
#
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f experiments/site.conf ] || { echo "copy experiments/site.conf.example to experiments/site.conf first" >&2; exit 1; }
. experiments/site.conf
job="$1"; shift
args=(--gres="gpu:${SMAT_GPUS:-1}" --time="${SMAT_TIME:-02:00:00}"
      --cpus-per-task="${SMAT_CPUS:-8}" --mem="${SMAT_MEM:-64G}")
[ -n "${SMAT_PARTITION:-}" ] && args+=(--partition="$SMAT_PARTITION")
[ -n "${SMAT_ACCOUNT:-}"   ] && args+=(--account="$SMAT_ACCOUNT")
[ -n "${SMAT_EXCLUDE:-}"   ] && args+=(--exclude="$SMAT_EXCLUDE")
mkdir -p logs
exec sbatch "${args[@]}" "$job" "$@"

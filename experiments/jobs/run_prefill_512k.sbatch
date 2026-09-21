#!/bin/bash
#SBATCH --job-name=smat-prefill-512k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# The prefill sweep of results_bf16.csv, extended from 256K to 512K.
#
# At 256K, d=4 (551 ms) is still just above fused causal attention (502 ms);
# the question is whether it crosses at 512K, where softmax's quadratic cost
# should roughly quadruple while d=4's T^{2-3/d} = T^{1.25} long-range work
# grows more slowly.  The prime q steps 23 -> 29 on the way, so the local
# growth of d=4 is steeper than its asymptote and the crossing is not certain.
#
# The whole sweep, 4K through 512K, reruns here in one job rather than adding a
# 512K row to the old file: the scheduler cannot tell apart two memory variants
# of the same card, and a point measured on one with ~30% more bandwidth would
# not belong on the same curve.  It also records the p25/p75 spread the old sweep predates,
# which is what fig16's error bars need.
#
# Flags are results_bf16_config.json verbatim except T, baseline_max_T (SDPA
# up to 512K) and --out.  At 512K, d=4's I_prof is ~21M against the 40M cap and
# its peak memory ~15 GB, so nothing is skipped and it fits a 40 GB card.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# one file per card: a curve must come from one GPU, never two
OUT="${1:-results_bf16_512k}"
source experiments/prelude.sh

# Refuse to time the torch fallback.  After the reorg a stale import left
# have_triton() False with no error, and a sweep ran on the torch path; these
# numbers would not be comparable with results_bf16.csv, which used the kernels.
python - <<'PY'
import torch, smat.attention as a
c = torch.cuda.get_device_capability(0)       # capability, not a product name
print("device sm_%d%d  %.0f GiB" % (*c, torch.cuda.get_device_properties(0).total_memory / 2**30))
assert a.have_triton(), "Triton kernels unavailable: refusing to time the torch path"
print("triton kernels: available")
PY

python -u analysis/bench_prefill.py \
    --device cuda --dtype bf16 \
    --T 4096 8192 16384 32768 65536 131072 262144 524288 \
    --d 1 2 3 4 \
    --nb 8 --r 64 --d-qk 64 --d-v 64 --chunk 128 --n-P 0 \
    --max-I-prof 40000000 --max-I-prof-torch 2000000 --max-I-tok 20000000 \
    --baseline-max-T 524288 --dense-max-T 4096 --numeric-max-T 16384 \
    --decode-steps 64 --warmup 5 --repeat 15 --seed 0 \
    --out "results/${OUT}.csv"

# a kernel that failed mid-sweep falls back per step with a WARNING; make it loud
if grep -q "WARNING: triton kernel" "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null; then
    echo "a Triton kernel fell back to torch during the sweep: timings are mixed"
    exit 1
fi
echo "done: $(date)"

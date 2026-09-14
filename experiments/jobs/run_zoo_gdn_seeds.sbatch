#!/bin/bash
#SBATCH --job-name=zoo-gdn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Gated DeltaNet base (fla 0.5.2 GatedDeltaNet, head_dim 16, expand_v 1 -> 16x16 state/head, 2*d_model/16 heads):
# seeds 123,1,2,3,4 x {GDN control (d=1), SMAT-GDN d=2 content read, SMAT-GDN d=3 content read} at width $WIDTH,
# lr 1e-2, 32 epochs; 15 runs spread over the node's 4 GPUs.   -> logs/zoo-gdn-{ctrl,d2,d3}-dm{W}-s{S}.out
# Needs triton >= 3.7.1 for fla's Hopper backward (isolated in ~/triton371) and FLA_TILELANG=0.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/triton371:/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:$PYTHONPATH
export FLA_TILELANG=0 WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_BASE=gdn ZOO_HEADDIM=16 ZOO_LAMACT=sigmoid ZOO_READ=content ZOO_LRS=1e-2
WIDTH="${WIDTH:-16}"
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba\|raw_cnt\|torchao" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
python -c "import triton; print('triton', triton.__version__)"; nvidia-smi -L
i=0
for seed in 123 1 2 3 4; do for arm in ctrl d2 d3; do
  case $arm in ctrl) ds=1;; d2) ds=2;; d3) ds=3;; esac
  gpu=$((i % 4)); i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu ZOO_DM=$WIDTH ZOO_DS=$ds ZOO_SEED=$seed python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$FILT" > logs/zoo-gdn-$arm-dm$WIDTH-s$seed.out 2>&1 ) &
  sleep 2
done; done
echo "launched $i runs over 4 GPUs: $(date)"
wait
echo "done: $(date)"

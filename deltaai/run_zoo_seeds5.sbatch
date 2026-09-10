#!/bin/bash
#SBATCH --job-name=zoo-seeds5
#SBATCH --account=bekw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/zoo-seeds5-%j.out

# Seeds 1-4 (seed 123 already run) x widths $WIDTHS x {Mamba-2 control, SMAT d=2 content read, SMAT d=3 content read}
# at hd16 / ds16 / lr 1e-2 / 32 epochs.  12 runs per width, spread round-robin over the node's 4 GPUs.
#   -> logs/zoo-seed5-{ctrl,d2,d3}-dm{W}-s{S}.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LAMACT=sigmoid ZOO_POOL=delta ZOO_READ=content ZOO_LRS=1e-2
WIDTHS="${WIDTHS:-16 32}"
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
nvidia-smi -L
i=0
for dm in $WIDTHS; do for seed in 1 2 3 4; do for arm in ctrl d2 d3; do
  case $arm in ctrl) ds=1;; d2) ds=2;; d3) ds=3;; esac
  gpu=$((i % 4)); i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu ZOO_DM=$dm ZOO_DS=$ds ZOO_SEED=$seed python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-seed5-$arm-dm$dm-s$seed.out 2>&1 ) &
  sleep 2
done; done; done
echo "launched $i runs over 4 GPUs: $(date)"
wait
echo "done: $(date)"

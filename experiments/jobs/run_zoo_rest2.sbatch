#!/bin/bash
#SBATCH --job-name=zoo-rest2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Zoology pipeline, width 16, SMAT mixer built on Zoology's own Mamba2 class
# (reset by folding the two equal segments into the batch; G output zero-init).
# Order: smoke -> d=1 control (1 lr) -> hybrid d=2 (4 lrs) -> Mamba-2 remaining
# lrs -> with-reset d=2, d=3.
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/_prelude.sh"
export PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_DM=16
F='grep --line-buffered -vi warn | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
echo "==== smoke ===="
ZOO_SMOKE=1 ZOO_EPOCHS=1 ZOO_DS=2,1 ZOO_RESET=1 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$F"
ZOO_SMOKE=1 ZOO_EPOCHS=1 ZOO_DS=2   ZOO_RESET=0 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$F"
echo "==== d=1 control, lr 1e-3 ===="
ZOO_EPOCHS=32 ZOO_DS=1 python - <<'PY' 2>&1 | eval "$F"
import importlib.util
spec = importlib.util.spec_from_file_location("cfg", "experiments/zoology/smat.py"); cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
from zoology.train import train
train([c for c in cfg.configs if abs(c.learning_rate - 1e-3) < 1e-9][0])
PY
echo "==== hybrid d=2 (no reset), 4 lrs ===="
ZOO_EPOCHS=32 ZOO_DS=2 ZOO_RESET=0 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$F"
echo "==== mamba2, remaining lrs ===="
ZOO_EPOCHS=32 ZOO_MODELS=mamba2 python - <<'PY' 2>&1 | eval "$F"
import importlib.util
spec = importlib.util.spec_from_file_location("cfg", "experiments/zoology/reference.py"); cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
from zoology.train import train
for c in cfg.configs:
    if c.learning_rate > 2e-3: train(c)
PY
echo "==== with-reset d=2, d=3, 4 lrs ===="
ZOO_EPOCHS=32 ZOO_DS=2,3 ZOO_RESET=1 python -m zoology.launch experiments/zoology/smat.py 2>&1 | eval "$F"
echo "done: $(date)"

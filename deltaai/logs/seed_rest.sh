#!/bin/bash
# three remaining seed runs inside ONE step (per-job step cap hit): ctrl s1, mr3 s1, mr3 s3
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_DM=16 ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_HEADDIM=8 ZOO_LAMACT=sigmoid ZOO_LRS=1e-2
cd /u/archerdw/smat_attention/deltaai
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
( ZOO_DS=1 ZOO_SEED=1 python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-seed-ctrl-hd8-s1.out 2>&1 ) &
( ZOO_DS=2 ZOO_SEED=1 python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-seed-mr3-hd8-s1.out 2>&1 ) &
( ZOO_DS=2 ZOO_SEED=3 python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-seed-mr3-hd8-s3.out 2>&1 ) &
wait

#!/bin/bash
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
export PYTHONPATH=/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_HEADDIM=16 ZOO_DSTATE=16 ZOO_LAMACT=sigmoid ZOO_POOL=delta ZOO_READ=content ZOO_DS=2 ZOO_LRS=1e-2
cd /u/archerdw/smat_attention/deltaai
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
for dm in 16 32 64; do
  ( ZOO_DM=$dm python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-ds16-cread-dm$dm.out 2>&1 ) &
done
wait

#!/bin/bash
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
export PYTHONPATH=/u/archerdw/triton371:/u/archerdw/smat_attention/deltaai:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:$PYTHONPATH
export FLA_TILELANG=0 WANDB_MODE=disabled ZOO_EPOCHS=32 ZOO_MR3=1 ZOO_BASE=gdn ZOO_HEADDIM=16 ZOO_LAMACT=sigmoid ZOO_READ=content ZOO_LRS=1e-2 ZOO_GDN_HEADS=1 ZOO_GDN_EXPANDV=1 ZOO_DM=16
cd /u/archerdw/smat_attention/deltaai
FILT='grep --line-buffered -vi warn | grep --line-buffered -v "hi mamba\|raw_cnt\|torchao" | grep --line-buffered -E "run_id=|valid/accuracy=|Traceback|Error|early" | sed -u -E "s/.*(valid\/loss=.*)\]$/\1/" | awk "!seen[\$0]++ {print; fflush()}"'
i=0
for seed in 123 1 2 3 4; do for arm in ctrl d2 d3; do
  case $arm in ctrl) ds=1;; d2) ds=2;; d3) ds=3;; esac
  gpu=$((i % 4)); i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu ZOO_DS=$ds ZOO_SEED=$seed python -m zoology.launch zoo_smat_configs.py 2>&1 | eval "$FILT" > logs/zoo-gdnm-$arm-dm16-s$seed.out 2>&1 ) &
  sleep 2
done; done
echo "launched $i"; wait

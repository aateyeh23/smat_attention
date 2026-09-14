#!/bin/bash
set -eo pipefail
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
cd /u/archerdw/smat_attention/deltaai
export PYTHONPATH="/u/archerdw/triton371:$PWD:/u/archerdw/.local/lib/python3.12/site-packages:/u/archerdw/zoology:${PYTHONPATH:-}"
export TRITON_F32_DEFAULT=tf32x3
export FLA_TILELANG=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled
export TRITON_CACHE_DIR="$PWD/results/mqar_gdn_reset/check_cache"
mkdir -p results/sc_gdn_smat/check
for d in 1 2 3 4; do
 arm=gdn_smat
 if [ "$d" = 1 ]; then arm=gdn; fi
 python -u lm/sc_gdn_train.py --hash_src ssm --lam_act sigmoid --arm "$arm" --d "$d" --gdn_headdim 16 --headdim 64 --d_state 16 --d_model 64 --n_layers 2 --seq_len 4096 --batch 2 --steps 1 --eval_every 1 --ckpt_every 1 --log_every 1 --anneal 1000 --warmup 500 --ckpt "results/sc_gdn_smat/check/ssm-d$d.pt" > "results/sc_gdn_smat/check/ssm-d$d.log" 2>&1
 echo "PASS SC d=$d full-length training/evaluation/checkpoint"
done
python -u lm/sc_gdn_train.py --hash_src ssm --lam_act sigmoid --arm gdn_smat --d 3 --gdn_headdim 16 --headdim 64 --d_state 16 --d_model 64 --n_layers 2 --seq_len 4096 --batch 2 --steps 2 --eval_every 1 --ckpt_every 1 --log_every 1 --anneal 1000 --warmup 500 --ckpt results/sc_gdn_smat/check/ssm-d3.pt >> results/sc_gdn_smat/check/ssm-d3.log 2>&1
echo 'PASS SC checkpoint resume'

python -u test_gdn_heads.py > results/sc_gdn_smat/check/alignment.log 2>&1
python -u test_delta_shards.py > results/sc_gdn_smat/check/shards.log 2>&1
touch results/sc_gdn_smat/check/validated

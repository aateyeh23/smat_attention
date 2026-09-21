#!/bin/bash
# When the current PG-19 slot (job $1) ends, wait for the next pg19 job to be RUNNING and resume the five
# selective-copying tweak arms on its GPU (they checkpoint every 250 steps).
OLD=$1; cd /u/an author/smat_attention/the GPU cluster
while squeue -h -j $OLD -o "%T" 2>/dev/null | grep -q RUNNING; do sleep 30; done
for i in $(seq 1 240); do NEW=$(squeue -u an author -h -n pg19 -t RUNNING -o "%i" | head -1); [ -n "$NEW" ] && break; sleep 30; done
[ -z "$NEW" ] && { echo "no new pg19 job"; exit 1; }
sleep 90; echo "resuming on job $NEW $(date +%H:%M)"
BASE="--arm smat --reset 0 --seq_len 4096 --n_copy 16 --n_vocab 16 --d_model 64 --n_layers 2 --headdim 64 --d_state 16 --batch 32 --steps 10000 --warmup 500 --lr 1e-3 --hash_conv_width 4 --eval_every 250 --ckpt_every 250 --log_every 50 --max_minutes 105"
ENVS="PYTHONPATH=/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
launch() { name=$1; extra=$2; grep -q "FINAL\|SOLVED" logs/sc4k-tweak-$name.out 2>/dev/null && return; E2=""; case "$extra" in *"--arm gdn"*) E2="PYTHONPATH=/u/an author/triton371:/u/an author/smat_attention/the GPU cluster:/u/an author/.local/lib/python3.12/site-packages:/u/an author/zoology FLA_TILELANG=0";; esac; nohup srun --jobid=$NEW --overlap --network=no_vni -N1 -n1 --export=ALL bash -c "module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1; cd /u/an author/smat_attention/the GPU cluster; env $ENVS $E2 python -u tasks/copying.py $BASE $extra --ckpt ${SMAT_WORK}/ckpt/sc4k-tweak-$name.pt 2>&1 | grep --line-buffered -v -i 'warn\|hi mamba\|custom_\|torchao'" < /dev/null >> logs/sc4k-tweak-$name.out 2>&1 & sleep 2; }
launch d3-goff "--d 3 --lam_act zero"
launch d3-goff-s1 "--d 3 --lam_act zero --seed 1"
launch d3-goff-s2 "--d 3 --lam_act zero --seed 2"
launch d3-gate-s1 "--d 3 --lam_act sigmoid --anneal 1000 --balance 0.01 --balance_gated 1 --seed 1"
launch d3-gate-s2 "--d 3 --lam_act sigmoid --anneal 1000 --balance 0.01 --balance_gated 1 --seed 2"
launch d3-reset-gate-ssm "--d 3 --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1 --reset 1"
launch d3-gate "--d 3 --lam_act sigmoid --anneal 1000 --balance 0.01 --balance_gated 1"
launch d3-gate-ssm "--d 3 --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1"
launch d3-ssm "--d 3 --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1"
launch d4-gate-ssm "--d 4 --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1"
launch d4-gate-ssm-s1 "--d 4 --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1 --seed 1"
launch d4-gate-ssm-s2 "--d 4 --lam_act sigmoid --hash_src ssm --anneal 1000 --balance 0.01 --balance_gated 1 --seed 2"
launch gdn-s1 "--arm gdn --gdn_headdim 16 --seed 1"
launch gdn-s2 "--arm gdn --gdn_headdim 16 --seed 2"
wait; echo "resume batch ended $(date +%H:%M)"

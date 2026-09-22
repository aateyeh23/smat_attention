#!/bin/bash
cd /u/archerdw/smat_attention/deltaai
for w in 16 32 64; do
  until [ "$(squeue -u archerdw -h | wc -l)" -lt 2 ]; do sleep 60; done
  out=$(sbatch --export=ALL,WIDTH=$w --job-name=zoo-gdnm-w$w run_zoo_gdnm_seeds.sbatch 2>&1); echo "$(date) width $w: $out"; sleep 30
done; echo "$(date) all submitted"

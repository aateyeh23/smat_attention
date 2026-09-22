"""Sequential real-training pilot, sharing one interactive GPU."""
import argparse
from pathlib import Path
import gc
import json
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--max-minutes', type=float, default=110)
p.add_argument('--completion-tag', default='')
p.add_argument('--families', default='mamba2,gdn_current')
p.add_argument('--task', default='state')
p.add_argument('--stage', default='tiny')
p.add_argument('--batch', type=int, default=32)
p.add_argument('--lr', type=float, default=.01)
p.add_argument('--eval-every', type=int, default=50)
p.add_argument('--eval-examples', type=int, default=256)
p.add_argument('--test-examples', type=int, default=1024)
p.add_argument('--steps', type=int, default=200)
p.add_argument('--min-length', type=int, default=256)
p.add_argument('--width', type=int, default=32)
p.add_argument('--loads', default='16,64')
p.add_argument('--ds', default='1,3')
p.add_argument('--seeds', default='123')
a = p.parse_args()
root = Path(__file__).resolve().parent/'results'/'synthetic_memory'/a.stage
started = time.monotonic()
for load in map(int, a.loads.split(',')):
    for seed in map(int, a.seeds.split(',')):
        for d in map(int, a.ds.split(',')):
            for family in a.families.split(','):
                folder = root/a.task/f'w{a.width}-n{load}-{family}-d{d}-s{seed}'
                if (folder/'result.json').exists() and 'binding_gap_pp' in json.loads((folder/'result.json').read_text()): continue
                minutes = a.max_minutes-(time.monotonic()-started)/60
                if minutes < 3: sys.exit(0)
                cmd = ['--task', a.task,
                    '--load', str(load), '--seed', str(seed), '--family', family,
                    '--d', str(d), '--width', str(a.width), '--steps', str(a.steps),
                    '--max-minutes', str(minutes), '--min-length', str(a.min_length),
                    '--batch', str(a.batch), '--lr', str(a.lr), '--eval-every', str(a.eval_every),
                    '--eval-examples', str(a.eval_examples), '--test-examples', str(a.test_examples),
                    '--out', str(folder)]
                from synthetic_memory import run, argument_parser
                run(argument_parser().parse_args(cmd))
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                if not (folder/'result.json').exists(): sys.exit(0)
(root/a.task/('COMPLETE'+a.completion_tag)).write_text('All requested cells completed.\n')

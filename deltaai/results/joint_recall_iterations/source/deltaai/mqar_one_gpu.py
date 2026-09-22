"""Run the MQAR campaign with bounded concurrency and resumable time slices."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
GROUPS = {
    'softroutes': ('zoo_gdn_soft_routes_configs.py', ROOT/'results/mqar_gdn_soft_routes/quoted_heads_s123'),
    'routegrad': ('zoo_gdn_route_grad_configs.py', ROOT/'results/mqar_gdn_route_grad/quoted_heads_s123'),
    'causalsoft': ('zoo_gdn_causal_soft_configs.py', ROOT/'results/mqar_gdn_causal_soft/quoted_heads_s123'),
    'causalkeys': ('zoo_gdn_causal_keys_configs.py', ROOT/'results/mqar_gdn_causal_keys/quoted_heads_s123'),
    'tiedkeys': ('zoo_gdn_tied_keys_configs.py', ROOT/'results/mqar_gdn_tied_keys/quoted_heads_s123'),
    'alignedsoft': ('zoo_gdn_aligned_soft_configs.py', ROOT/'results/mqar_gdn_aligned_soft/quoted_heads_s123'),
    'aligned': ('zoo_gdn_aligned_configs.py', ROOT/'results/mqar_gdn_aligned/quoted_heads_s123'),
    'scgdn': ('lm/sc_gdn_train.py', ROOT/'results/sc_gdn_smat/s0'),
    'delta': ('zoo_gdn_delta_configs.py', ROOT/'results/mqar_gdn_delta/quoted_heads_s123'),
    'gdn': ('zoo_gdn_reset_configs.py', ROOT/'results/mqar_gdn_reset/quoted_heads_s123'),
    'mamba': ('zoo_mamba_reset_configs.py', ROOT/'results/mqar_mamba_reset/w16_hd16_ds16_independent_s123'),
    'interleaved': ('zoo_gdn_interleaved_configs.py', ROOT/'results/mqar_gdn_interleaved/quoted_heads_s123'),
}
# Establish every baseline and interleave both tasks before broader ablations.
TASKS = [('interleaved',32,1), ('aligned',32,3),
         ('interleaved',16,1), ('interleaved',64,1), ('scgdn',64,1),
         ('tiedkeys',32,3),
         ('causalkeys',32,3), ('causalsoft',32,3), ('softroutes',32,3), ('softroutes',16,3), ('softroutes',64,3), ('causalkeys',16,3), ('causalkeys',64,3),
         ('scgdn',64,3),
         ('scgdn',64,2), ('scgdn',64,4),
         ('mamba',16,1), ('mamba',16,2), ('mamba',16,3), ('mamba',16,4),
         ]
# Older hard-routing GDN candidates retired after the soft-routing diagnosis.


def paths(task):
    group,width,d = task
    config,folder = GROUPS[group]
    return config,folder,folder/f'w{width}-d{d}.json'


def complete(task):
    _,_,status = paths(task)
    return status.exists() and json.loads(status.read_text())['complete']


def main():
    pending = [t for t in TASKS if not complete(t)]
    if not pending:
        print('ALL MQAR RUNS COMPLETE', flush=True)
        return 0
    deadline = time.monotonic() + float(os.environ.get('MQAR_SLICE_MINUTES','100'))*60
    active = []
    failed = False
    concurrency = int(os.environ.get('MQAR_CONCURRENCY','4'))
    while pending or active:
        remaining = (deadline-time.monotonic())/60
        while pending and len(active)<concurrency and remaining>5:
            task = pending.pop(0)
            config,folder,_ = paths(task)
            folder.mkdir(parents=True,exist_ok=True)
            group,width,d = task
            env = os.environ.copy()
            env.update(ZOO_DM=str(width), ZOO_DS=str(d), ZOO_EPOCHS='32',
                       MQAR_RUN_DIR=str(folder), MQAR_MAX_MINUTES=str(max(1,remaining-3)))
            log = (folder/f'w{width}-d{d}.log').open('a')
            process = subprocess.Popen([sys.executable,'-u','-m','zoology.launch',config],
                                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append((task,process,log))
            print(f'START {task} pid={process.pid} budget_minutes={remaining-3:.1f}',flush=True)
        for item in active[:]:
            task,process,log = item
            code = process.poll()
            if code is None:
                continue
            log.close();active.remove(item)
            print(f'EXIT {task} code={code} complete={complete(task)}',flush=True)
            failed |= code != 0
        if not active and (failed or remaining<=5):
            break
        if failed:
            pending.clear()  # Let the other running arms finish/checkpoint, then stop for diagnosis.
        if active:
            time.sleep(2)
    summary = {'complete':[list(t) for t in TASKS if complete(t)],
               'unfinished':[list(t) for t in TASKS if not complete(t)], 'failed':failed}
    status = ROOT/'results/mqar_one_gpu/status.json'
    status.write_text(json.dumps(summary,indent=2))
    if failed:
        return 1
    return 0 if not summary['unfinished'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

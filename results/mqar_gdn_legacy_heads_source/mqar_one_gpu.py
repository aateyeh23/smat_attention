"""Run the MQAR campaign with bounded concurrency and resumable time slices."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
GROUPS = {
    'delta': ('zoo_gdn_delta_configs.py', ROOT/'results/mqar_gdn_delta/w32_s123'),
    'gdn': ('zoo_gdn_reset_configs.py', ROOT/'results/mqar_gdn_reset/w16w32_s123'),
    'mamba': ('zoo_mamba_reset_configs.py', ROOT/'results/mqar_mamba_reset/w16_hd16_ds16_independent_s123'),
    'interleaved': ('zoo_gdn_interleaved_configs.py', ROOT/'results/mqar_gdn_interleaved/w16w32_s123'),
}
# Corrected width-16 Mamba arms first, followed by the existing GDN campaign.
TASKS = [('delta',32,3), ('mamba',16,1), ('mamba',16,2), ('mamba',16,3), ('mamba',16,4),
         ('interleaved',16,1), ('interleaved',32,1),
         ('interleaved',16,3), ('interleaved',32,3)]
TASKS += [('gdn',w,d) for d in (1,3,2,4) for w in (16,32,64)]


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

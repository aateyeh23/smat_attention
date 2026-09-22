"""One process per GPU; resume checkpoints, never stop based on accuracy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent/'results/paper_seeds_20260920'


def environment(task, device):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PAPER_TASK=json.dumps(task))
    env['TRITON_CACHE_DIR'] = f'/tmp/{os.environ["USER"]}-paper-seeds-triton'
    source = ROOT/'source'
    if task['task'] == 'mqar':
        base = source/'mqar'
        paths = [base/'deltaai', base/'smat', base]
        if task['family'] == 'gdn' and task['width'] >= 32 and task['d'] >= 2:
            paths.insert(0, source/'transport')
        folder = Path(task['result']).parent
        env.update(MQAR_FAMILY=task['family'], ZOO_DM=str(task['width']),
                   ZOO_DS=str(max(1, task['d'])), ZOO_EPOCHS='32', MQAR_RUN_DIR=str(folder))
        script = CODE/'run_mqar.py'
        args = []
    else:
        base = source/'joint'
        paths = [base/'deltaai', base/'smat', base]
        script = base/'deltaai/joint_recall.py'
        dataset = ROOT/'joint'
        if task['task'] == 'joint_loglinear':
            paths.insert(0, CODE)
            paths.insert(0, ROOT/'python_deps')
            script = CODE/'run_joint_loglinear.py'
            dataset = ROOT/'joint_loglinear'
        args = ['--root', str(dataset), '--seeds', str(task['seed']), '--conditions', 'shared',
                '--families', task['family'], '--ds', str(task['d']), '--lr', str(task['lr'])]
    env['PYTHONPATH'] = ':'.join(map(str, [Path('/u/archerdw/triton371'), *paths,
                                       Path('/u/archerdw/.local/lib/python3.12/site-packages')]))
    env.pop('ZOO_SMOKE', None)
    return env, [sys.executable, '-P', '-u', str(script), *args]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=float, default=105)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--only', help='Run one explicitly assigned dedicated task')
    args = parser.parse_args()
    for relative, expected in json.loads((ROOT/'source_sha256.json').read_text()).items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected, relative
    tasks = json.loads((ROOT/'tasks.json').read_text())
    if args.only:
        tasks = [t for t in tasks if t['name'] == args.only]
        assert len(tasks) == 1
    else:
        tasks = [t for t in tasks[args.shard::args.shards] if not t.get('dedicated')]
    status_name = f'dedicated-{args.only}' if args.only else str(args.shard)
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    deadline = time.monotonic()+args.minutes*60
    running = {}; failed = []; attempted = set()
    logs = ROOT/'logs'; logs.mkdir(exist_ok=True)
    while True:
        for device, (proc, task, log) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close(); del running[device]
            print('FINISH', task['name'], rc, flush=True)
            if rc:
                failed.append(dict(name=task['name'], exit_code=rc))
        minutes = (deadline-time.monotonic())/60
        pending = [t for t in tasks if t['name'] not in attempted and not Path(t['result']).exists()
                   and ('gate' not in t or (ROOT/t['gate']).exists())]
        for device in devices:
            if device in running or not pending or minutes < 5 or failed:
                continue
            task = pending.pop(0); attempted.add(task['name'])
            env, command = environment(task, device)
            env['MQAR_MAX_MINUTES'] = str(max(1, minutes-5))
            if task['task'] != 'mqar':
                command += ['--max-minutes', str(minutes)]
            log = (logs/f"{os.environ.get('SLURM_JOB_ID','local')}-{task['name']}.out").open('w')
            running[device] = (subprocess.Popen(command, env=env, stdout=log,
                                                stderr=subprocess.STDOUT), task, log)
            print('START', device, task['name'], flush=True)
        status = dict(job=os.environ.get('SLURM_JOB_ID'), total=len(tasks),
                      completed=[t['name'] for t in tasks if Path(t['result']).exists()],
                      running=[t['name'] for _, t, _ in running.values()], failed=failed)
        temporary = ROOT/f'status-{status_name}.tmp'; temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(ROOT/f'status-{status_name}.json')
        if not running:
            break
        time.sleep(5)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

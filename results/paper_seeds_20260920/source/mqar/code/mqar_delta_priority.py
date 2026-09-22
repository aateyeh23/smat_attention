"""Give one urgent experiment a bounded, exclusive training slice in a live job.

Suspend the campaign worker and its current training child; always resume them
when this slice exits. The experiment's normal epoch checkpoints are reusable by
the campaign. This script itself runs as a Slurm step, not a login background job.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from mqar_dispatch import STATE, claim, release, paths, write


def children(pid):
    return [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]


def resume(pids):
    for pid in reversed(pids):
        try: os.kill(pid, signal.SIGCONT)
        except ProcessLookupError: pass


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, interrupted)
task = ['delta', 32, 3]
job = os.environ['SLURM_JOB_ID']
paused = []
directory = None
training = None
owner_path = None
original_owner = None
try:
    # Freeze the scheduler first to prevent it starting another child in the gap.
    owners = [json.loads(p.read_text()) for p in (STATE/'claims').glob('*/owner.json')]
    own = [o for o in owners if o['job'] == job]
    if len(own) != 1: raise RuntimeError(f'Expected exactly one active campaign task: {own}')
    worker = own[0]['pid']
    os.kill(worker, signal.SIGSTOP); paused.append(worker)
    for pid in children(worker):
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ')
        if b'zoology.launch' in command:
            os.kill(pid, signal.SIGSTOP); paused.append(pid)
    if len(paused) != 2: raise RuntimeError(f'Expected one training child: {paused}')
    from mqar_dispatch import key
    original_owner = own[0]
    owner_path = STATE/'claims'/key(original_owner['task'])/'owner.json'
    write(owner_path, dict(original_owner, paused_for=task))
    directory = claim(task,job)
    if directory is None: raise RuntimeError('Delta task already claimed')
    config,folder,_ = paths(task)
    folder.mkdir(parents=True,exist_ok=True)
    write(folder/'priority.json',dict(job=job,paused=paused,started=time.time(),status='running'))
    env=os.environ.copy()
    env.pop('ZOO_SMOKE', None)
    env.update(ZOO_DM='32',ZOO_DS='3',ZOO_EPOCHS='32',MQAR_RUN_DIR=str(folder),MQAR_MAX_MINUTES='25')
    print(f'PRIORITY START {task}; suspended {own[0]["task"]}, pids={paused}',flush=True)
    with (folder/'w32-d3.log').open('a') as log:
        training=subprocess.Popen([sys.executable,'-u','-m','zoology.launch',config],env=env,stdout=log,stderr=subprocess.STDOUT)
        code=training.wait()
    write(folder/'priority.json',dict(job=job,paused=paused,ended=time.time(),status='finished',exit_code=code))
    if code:
        write(STATE/'errors'/'delta-32-3.json',dict(task=task,job=job,exit_code=code))
    raise SystemExit(code)
finally:
    if training is not None and training.poll() is None:
        training.terminate()
        try: training.wait(timeout=20)
        except subprocess.TimeoutExpired: training.kill();training.wait()
    if directory is not None: release(directory)
    if owner_path is not None and owner_path.exists(): write(owner_path, original_owner)
    resume(paused)
    print('PRIORITY END: prior campaign worker resumed',flush=True)

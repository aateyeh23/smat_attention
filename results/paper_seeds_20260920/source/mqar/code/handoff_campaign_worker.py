"""Adopt the refreshed task list after the current trainer exits, without killing it."""
import json,os,signal,subprocess,time
from pathlib import Path
from mqar_dispatch import STATE, worker, write


def interrupted(signum,frame): raise SystemExit(128+signum)
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,interrupted)
job=os.environ['SLURM_JOB_ID']
owners=[json.loads(p.read_text()) for p in (STATE/'claims').glob('*/owner.json')]
active=[o for o in owners if o['job']==job]
assert len(active)==1,active
old=active[0]['pid']
assert b'mqar_dispatch.py' in Path(f'/proc/{old}/cmdline').read_bytes()
paused=False
try:
    os.kill(old,signal.SIGSTOP);paused=True
    children=[int(x) for x in Path(f'/proc/{old}/task/{old}/children').read_text().split()]
    trainers=[p for p in children if b'zoology.launch' in Path(f'/proc/{p}/cmdline').read_bytes()]
    assert len(trainers)==1,trainers
    child=trainers[0]
    print(f'Waiting for current trainer {child} to finish; scheduler {old} paused',flush=True)
    while Path(f'/proc/{child}/stat').exists():
        if Path(f'/proc/{child}/stat').read_text().split()[2]=='Z': break
        time.sleep(5)
    elapsed=subprocess.check_output(['squeue','-h','-j',job,'-o','%M'],text=True).strip()
    fields=[int(x) for x in elapsed.split(':')]
    minutes=fields[-2]+fields[-1]/60+(fields[-3]*60 if len(fields)==3 else 0)
    budget=max(0,95-minutes)
    os.environ['MQAR_FIRST_TASK']='aligned-32-3'
    print(f'Fresh campaign worker starts with budget {budget:.1f} minutes',flush=True)
    code=worker('bulk',None,budget)
    raise SystemExit(code)
finally:
    if paused:
        try: os.kill(old,signal.SIGCONT)
        except ProcessLookupError: pass
    print('Original allocation worker resumed',flush=True)

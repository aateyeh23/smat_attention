"""Sequential interactive worker plus a refreshed multi-node backlog allocation."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import importlib
campaign = importlib.import_module(os.environ.get('MQAR_CAMPAIGN_MODULE', 'mqar_one_gpu'))
TASKS, paths = campaign.TASKS, campaign.paths

ROOT=Path(__file__).resolve().parent
STATE=Path(os.environ.get('MQAR_DISPATCH_STATE',ROOT/'results/mqar_dispatch'))
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'}
JOB_NAMES={}
JOB_PREFIX=os.environ.get('MQAR_JOB_PREFIX', 'mqar')


def key(task): return '-'.join(map(str,task))
def refresh_campaign():
    updated=importlib.reload(campaign)
    TASKS[:] = updated.TASKS
def read(path,default=None):
    try: return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return default

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)

def done(task):
    status=read(paths(task)[2],{})
    return bool(status.get('complete'))

def claim(task,job):
    directory=STATE/'claims'/key(task)
    directory.parent.mkdir(parents=True,exist_ok=True)
    try: directory.mkdir()
    except FileExistsError: return None
    write(directory/'owner.json',dict(task=task,job=str(job),pid=os.getpid(),host=socket.gethostname()))
    return directory

def release(directory): shutil.rmtree(directory)

def commands(args):
    return subprocess.check_output(args,cwd=ROOT,text=True).strip()

def jobs():
    output=commands(['squeue','-h','-u',os.environ['USER'],'-o','%i|%T|%j'])
    rows=[line.split('|',2) for line in output.splitlines() if line]
    JOB_NAMES.clear();JOB_NAMES.update({job:name for job,_,name in rows})
    return {job:state for job,state,_ in rows}

def owners(current):
    result={}
    for directory in (STATE/'claims').glob('*'):
        owner=read(directory/'owner.json')
        if owner is None:
            if time.time()-directory.stat().st_mtime>300: shutil.rmtree(directory)
        elif owner['job'] not in current:
            shutil.rmtree(directory)
        else: result[directory.name]=owner
    return result

def backlog(task_list,claimed,reserved=None):
    return [list(t) for t in task_list if not done(t) and key(t) not in claimed
            and (t[0]!='scgdn' or (ROOT/'results/sc_gdn_smat/check/validated').exists())
            and list(t)!=reserved and not (STATE/'errors'/f'{key(t)}.json').exists()]


def worker(role,manifest,minutes):
    candidates=read(Path(manifest)) if manifest else [list(t) for t in TASKS]
    if candidates is None: raise RuntimeError('Missing worker manifest')
    preferred=os.environ.get('MQAR_FIRST_TASK')
    if preferred:
        candidates.sort(key=lambda t:key(t)!=preferred)
    job=os.environ['SLURM_JOB_ID']
    # Honor shorter backfill allocations too, leaving time to save/exit.
    remaining=commands(['squeue','-h','-j',job,'-o','%L']).splitlines()[0]
    days,clock=(remaining.split('-',1) if '-' in remaining else ('0',remaining))
    fields=list(map(int,clock.split(':')))
    seconds=int(days)*86400+sum(v*60**i for i,v in enumerate(reversed(fields)))
    deadline=time.monotonic()+min(minutes*60,max(0,seconds-120))
    attempted=set()
    mamba_validated=False
    gdn_validated=False
    while deadline-time.monotonic()>300:
        if not manifest:
            refresh_campaign()
            candidates=[list(t) for t in TASKS]
            if preferred: candidates.sort(key=lambda t:key(t)!=preferred)
        selected=None
        for task in candidates:
            if task[0]=='scgdn' and not (ROOT/'results/sc_gdn_smat/check/validated').exists(): continue
            if key(task) in attempted or done(task) or (STATE/'errors'/f'{key(task)}.json').exists(): continue
            directory=claim(task,job)
            if directory:
                # A prior owner may have finished immediately before we claimed it.
                if done(task): release(directory);continue
                selected=(task,directory);break
        if selected is None: break
        task,directory=selected;attempted.add(key(task))
        config,folder,_=paths(task);folder.mkdir(parents=True,exist_ok=True)
        if hasattr(campaign, 'prepare_worker') and not gdn_validated:
            try:
                campaign.prepare_worker()
            except Exception as error:
                write(STATE/'errors'/f'{key(task)}.json',dict(task=task,job=job,preflight_failed=True,error=repr(error)))
                release(directory)
                raise
            gdn_validated=True
        if task[0] in ('gdn','delta','interleaved','aligned','alignedsoft','tiedkeys','causalkeys','causalsoft','routegrad','softroutes') and not gdn_validated:
            with (folder/f'heads-preflight-{job}-{os.getpid()}.log').open('w') as test_log:
                check=subprocess.run([sys.executable,'-u','test_gdn_heads.py'],cwd=ROOT,
                                     stdout=test_log,stderr=subprocess.STDOUT)
            if check.returncode:
                write(STATE/'errors'/f'{key(task)}.json',dict(task=task,job=job,preflight_failed=True))
                release(directory);return check.returncode
            gdn_validated=True
        if task[0]=='mamba' and not mamba_validated:
            with (folder/f'preflight-{job}-{os.getpid()}.log').open('w') as test_log:
                check=subprocess.run([sys.executable,'-u','test_mqar_mamba_reset.py'],cwd=ROOT,
                                     stdout=test_log,stderr=subprocess.STDOUT)
            if check.returncode:
                write(STATE/'errors'/f'{key(task)}.json',dict(task=task,job=job,preflight_failed=True))
                release(directory);return check.returncode
            mamba_validated=True
        env=os.environ.copy()
        env.update(ZOO_DM=str(task[1]),ZOO_DS=str(task[2]),ZOO_EPOCHS='32',
                   MQAR_RUN_DIR=str(folder),MQAR_MAX_MINUTES=str(max(1,(deadline-time.monotonic())/60-3)))
        print(f'START {role} {task} job={job}',flush=True)
        # Child inherits the allocation; a killed worker leaves the claim until Slurm ends the job.
        with (folder/f'w{task[1]}-d{task[2]}.log').open('a') as log:
            command=[sys.executable,'-u','-m','zoology.launch',config]
            if task[0]=='scgdn':
                env['TRITON_F32_DEFAULT']='tf32x3'
                command=[sys.executable,'-u',config,'--arm','gdn' if task[2]==1 else 'gdn_smat',
                    '--d',str(task[2]),'--d_model','64','--n_layers','2','--headdim','64',
                    '--d_state','16','--gdn_headdim','16','--seq_len','4096','--n_copy','16',
                    '--n_vocab','16','--batch','32','--steps','10000','--warmup','500',
                    '--lr','0.001','--wd','0.1','--anneal','1000','--balance','0.01',
                    '--hash_src','ssm','--lam_act','sigmoid',
                    '--eval_every','250','--ckpt_every','250','--log_every','50','--seed','0',
                    '--max_minutes',env['MQAR_MAX_MINUTES'],'--ckpt',str(folder/f'w{task[1]}-d{task[2]}.pt')]
            process=subprocess.Popen(command,cwd=ROOT,
                                     env=env,stdout=log,stderr=subprocess.STDOUT)
            code=process.wait()
        if code:
            write(STATE/'errors'/f'{key(task)}.json',dict(task=task,job=job,exit_code=code))
        release(directory)
        print(f'END {task} code={code} complete={done(task)}',flush=True)
        if code: return code
    return 0


def submit(role,manifest=None,first=None,count=0,dependency=None):
    cmd=['sbatch','--parsable','--comment=mqar-dispatch-v1']
    cmd += ['--job-name='+JOB_PREFIX+('-backlog' if role=='bulk' else '-sequential')]
    if dependency:cmd += ['--dependency=afterany:'+dependency]
    if role=='bulk':
        cmd += ['--nodes='+str(min(4,count)), 'run_mqar_dispatch_bulk.sbatch',str(manifest)]
    else:
        cmd += ['--export=ALL,MQAR_FIRST_TASK='+key(first),'run_mqar_dispatch_single.sbatch']
    job=commands(cmd).split(';')[0]
    print('SUBMITTED',role,job,flush=True)
    return job


def controller(interval):
    STATE.mkdir(parents=True,exist_ok=True)
    lock=(STATE/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(STATE/'controller.json',dict(pid=os.getpid(),host=socket.gethostname()))
    state=read(STATE/'jobs.json',{})
    while not (STATE/'STOP').exists():
        try:
            refresh_campaign()
            current=jobs();claimed=owners(current)
            if state.get('single') not in current and state.get('next_single') in current:
                state['single']=state.pop('next_single')
                state['reserved']=state.pop('next_reserved',None)
                write(STATE/'jobs.json',state)
            # Recover submitted allocations if the controller restarted between sbatch and saving its state.
            for role,name in [('single',JOB_PREFIX+'-sequential'),('bulk',JOB_PREFIX+'-backlog')]:
                candidates=[job for job in current if JOB_NAMES.get(job)==name]
                if state.get(role) not in current and candidates:
                    state[role]=candidates[0]
                    write(STATE/'jobs.json',state)
            unfinished=[list(t) for t in TASKS if not done(t)]
            if not unfinished:
                for role in ('single','next_single','bulk'):
                    job=state.get(role)
                    if current.get(job)=='PENDING': commands(['scancel','--state=PENDING',job])
                write(STATE/'status.json',dict(complete=True,unfinished=[]))
                release_job=getattr(campaign, 'RELEASE_JOB', '3143305')
                if release_job:
                    subprocess.run(['scontrol','update','JobId='+release_job,'Partition=ghx4-interactive','Nice=0'])
                    subprocess.run(['scontrol','release',release_job])
                print('ALL EXPERIMENTS COMPLETE',flush=True);return
            single=state.get('single')
            reserved=state.get('reserved') if current.get(single) in ('PENDING','CONFIGURING') else None
            available=backlog(TASKS,claimed)
            if single not in current and available:
                reserved=available[0]
                state['single']=submit('single',first=reserved)
                state['reserved']=reserved
                write(STATE/'jobs.json',state)
            # If a newly running single worker has not claimed yet, preserve its reservation.
            elif current.get(single)=='RUNNING' and not any(o['job']==single for o in claimed.values()):
                reserved=state.get('reserved')
            # A successor exists before the current allocation ends. Its worker restarts this controller.
            if current.get(single)=='RUNNING' and state.get('next_single') not in current:
                first=unfinished[0]
                state['next_single']=submit('single',first=first,dependency=single)
                state['next_reserved']=first
                write(STATE/'jobs.json',state)
            wanted=backlog(TASKS,claimed,reserved)
            bulk=state.get('bulk');bulk_state=current.get(bulk)
            if bulk_state=='PENDING' and wanted!=state.get('bulk_tasks'):
                # State-filtered cancellation avoids killing a batch that just started.
                commands(['scancel','--state=PENDING',bulk])
                refreshed=jobs()
                bulk_state=refreshed.get(bulk)
                if bulk_state=='PENDING':
                    time.sleep(1);bulk_state=jobs().get(bulk)
            if bulk_state not in ACTIVE and wanted:
                manifest=STATE/f'backlog-{time.time_ns()}.json';write(manifest,wanted)
                state['bulk']=submit('bulk',manifest=manifest,count=len(wanted))
                state['bulk_tasks']=wanted
                write(STATE/'jobs.json',state)
            elif bulk_state=='PENDING' and not wanted:
                commands(['scancel','--state=PENDING',bulk])
            write(STATE/'status.json',dict(complete=False,jobs=state,active_claims=claimed,
                                          unfinished=unfinished,errors=[str(p) for p in (STATE/'errors').glob('*.json')],
                                          updated=time.time()))
        except Exception as error:
            print('CONTROLLER ERROR',repr(error),flush=True)
        time.sleep(interval)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('role',choices=['controller','single','bulk'])
    parser.add_argument('--manifest');parser.add_argument('--minutes',type=float,default=100)
    parser.add_argument('--interval',type=float,default=20)
    args=parser.parse_args()
    if args.role=='controller':controller(args.interval)
    else:
        manager=None
        if args.role=='single':
            STATE.mkdir(parents=True,exist_ok=True)
            log=(STATE/'controller.log').open('a')
            manager=subprocess.Popen([sys.executable,'-u',__file__,'controller'],cwd=ROOT,
                                     stdout=log,stderr=subprocess.STDOUT)
        try:raise SystemExit(worker(args.role,args.manifest,args.minutes))
        finally:
            if manager is not None:
                if all(done(t) for t in TASKS):
                    try:manager.wait(timeout=30)
                    except subprocess.TimeoutExpired:manager.terminate();manager.wait()
                else:manager.terminate();manager.wait()
                log.close()

"""Run independent development comparisons across the GPUs in one allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--minutes',type=float,default=110);args=p.parse_args()
    tasks=json.loads(args.manifest.read_text())
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    started=time.monotonic();deadline=started+args.minutes*60
    running={};failed=[];attempted=set();logs=args.manifest.parent/'logs';logs.mkdir(exist_ok=True)
    while True:
        # A development iteration may append tasks or change pending priorities.
        # Existing subprocesses keep their immutable command and output root.
        tasks=json.loads(args.manifest.read_text())
        for device,(proc,task,log) in list(running.items()):
            rc=proc.poll()
            if rc is None:continue
            log.close();del running[device]
            print('FINISH',device,task['name'],'exit',rc,'complete',Path(task['result']).exists(),flush=True)
            if rc:failed.append(dict(task=task['name'],exit_code=rc))
        remaining=(deadline-time.monotonic())/60
        pending=[t for t in tasks if t['name'] not in attempted and not Path(t['result']).exists()]
        if remaining>3 and not failed:
            for device in devices:
                if device in running or not pending:continue
                task=pending.pop(0);attempted.add(task['name'])
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
                cmd=[sys.executable,'-u',task['script'],*task['args'],'--max-minutes',str(remaining)]
                logfile=logs/f'{os.environ.get("SLURM_JOB_ID","local")}-{task["name"]}.out'
                log=logfile.open('w')
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                running[device]=(proc,task,log)
                print('START',device,task['name'],'pid',proc.pid,'log',str(logfile),flush=True)
        status=dict(job=os.environ.get('SLURM_JOB_ID'),elapsed=time.monotonic()-started,
                    completed=[t['name'] for t in tasks if Path(t['result']).exists()],
                    running=[t['name'] for _,t,_ in running.values()],failed=failed,
                    total=len(tasks))
        temp=args.manifest.parent/'campaign_status.tmp';temp.write_text(json.dumps(status,indent=2)+'\n')
        temp.replace(args.manifest.parent/'campaign_status.json')
        if not running:break
        time.sleep(5)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()

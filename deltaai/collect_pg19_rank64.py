"""Collect lightweight live metrics for the four rank64 PG19 arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import modal
ROOT=Path(__file__).resolve().parent/'results/pg19_w768_rank64'
VOLUME='pg19-w768-rank64'


def collect(variant=""):
    output=ROOT/variant if variant else ROOT
    volume=modal.Volume.from_name(VOLUME)
    def fetch(name):
        root=output/name;root.mkdir(parents=True,exist_ok=True)
        row=dict(run=name,tokens=0,complete=False)
        for filename in ('manifest.json','recipe.json','status.json','metrics.jsonl','pilot.json'):
            try:data=b''.join(volume.read_file('/seed123/'+(variant+'/' if variant else '')+f'{name}/{filename}'))
            except (FileNotFoundError,modal.exception.NotFoundError):continue
            (root/filename).write_bytes(data)
            if filename=='status.json':
                saved=json.loads(data)
                row.update(saved);row['saved_tokens']=saved['tokens']
            if filename=='metrics.jsonl':
                for line in data.decode().splitlines():
                    try:metric=json.loads(line)
                    except json.JSONDecodeError:continue
                    row['tokens']=max(row['tokens'],metric['tokens'])
                    if metric['event'] in ('train','validation'):row[metric['event']]=metric
            if filename=='pilot.json':row['pilot']=json.loads(data)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(fetch,[f'{f}-d{d}' for f in ('gdn','mamba2') for d in (3,4)]))
    output.mkdir(parents=True,exist_ok=True)
    result=dict(updated=datetime.now(timezone.utc).isoformat(),runs=rows)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:print(json.dumps(row),flush=True)
    return all(row['complete'] for row in rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true')
    p.add_argument('--variant',default='fusedwrite-b2-ac0');args=p.parse_args()
    while True:
        done=collect(args.variant)
        if done or not args.watch:break
        time.sleep(60)

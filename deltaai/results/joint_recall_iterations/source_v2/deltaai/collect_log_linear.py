"""Download live MQAR Log-Linear metrics and completed checkpoints from Modal."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import modal
ROOT=Path(__file__).resolve().parent/'results/mqar_log_linear'
VOLUME='mqar-log-linear-s123'


def collect(family,width):
    name=f'{family}-w{width}'
    root=ROOT/name;root.mkdir(parents=True,exist_ok=True)
    volume=modal.Volume.from_name(VOLUME)
    stem=f'w{width}-d1'
    for file in ('manifest.json','optimizer-config.json',stem+'.json',stem+'-history.jsonl','result.json'):
        try: data=b''.join(volume.read_file('/'+name+'/'+file))
        except (FileNotFoundError,modal.exception.NotFoundError): continue
        (root/file).write_bytes(data)
    path=root/(stem+'.json')
    if not path.exists(): return dict(run=name,status='waiting for first checkpoint')
    status=json.loads(path.read_text())
    complete=bool(status['complete'])
    if complete and not (root/(stem+'.pt')).exists():
        tmp=root/(stem+'.pt.partial')
        with tmp.open('wb') as f:
            for block in volume.read_file('/'+name+'/'+stem+'.pt'): f.write(block)
        tmp.replace(root/(stem+'.pt'))
        (root/'train.log').write_bytes(b''.join(volume.read_file('/'+name+'/train.log')))
    return dict(run=name,epoch=status['next_epoch'],complete=complete,
                accuracy=status['metrics']['valid/accuracy'],metrics=status['metrics'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(collect,f,w) for f in ('gdn','mamba2') for w in (16,32,64)]
            rows=[f.result() for f in futures]
        (ROOT/'status.json').write_text(json.dumps(rows,indent=2)+'\n')
        for row in rows: print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
        if not args.watch or all(r.get('complete',False) for r in rows):break
        time.sleep(30)

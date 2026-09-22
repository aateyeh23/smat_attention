"""Collect epoch-gated candidates and preserve terminal checkpoints locally."""
import argparse
import json
from pathlib import Path
import time
import modal

ROOT=Path(__file__).resolve().parent/'results/mqar_gdn_iterations'


def collect(trial):
    out=ROOT/trial;out.mkdir(parents=True,exist_ok=True)
    volume=modal.Volume.from_name('smat-gdn-w32-d3-iterations')
    try: data=b''.join(volume.read_file('/'+trial+'/manifest.json'))
    except (FileNotFoundError,modal.exception.NotFoundError):
        print(trial+': waiting for launch',flush=True);return False
    (out/'manifest.json').write_bytes(data)
    manifest=json.loads(data)
    stem=f'w{manifest["width"]}-d{manifest["d"]}'
    for name in (stem+'.json',stem+'-history.jsonl','gates.jsonl',
                 'decision.json','result.json','gpu-validation.log','gradient-clipping.jsonl',
                 'optimizer-groups.json','optimizer-config.json','expected-sources.json',
                 'gpu-validation-sources.json','training-sources.json'):
        try: data=b''.join(volume.read_file('/'+trial+'/'+name))
        except (FileNotFoundError,modal.exception.NotFoundError): continue
        (out/name).write_bytes(data)
    path=out/(stem+'.json')
    if not path.exists():
        print(trial+': waiting for validation/first checkpoint',flush=True);return False
    try: status=json.loads(path.read_text())
    except json.JSONDecodeError: return False
    history=[json.loads(s) for s in (out/'gates.jsonl').read_text().splitlines()]
    report=[f'MQAR GDN + SMAT d={manifest["d"]}, width{manifest["width"]}: {trial}',
            '', manifest.get('gate', 'Strict gates at epochs 4,6,8,...,32; accuracy must exceed GDN.'),
            '', '| Epoch | Candidate | GDN | Margin (pp) | Decision |',
            '|---|---:|---:|---:|---|']
    for r in history:
        report.append(f'| {r["epoch"]} | {100*r["accuracy"]:.2f}% | {100*r["baseline"]:.2f}% | {100*r["margin"]:+.2f} | {r["decision"]} |')
    result_path=out/'result.json'
    done=result_path.exists()
    if done:
        result=json.loads(result_path.read_text())
        report.extend(['',f'Terminal result: {result["decision"]} at epoch {result["epoch"]}.'])
        if not (out/(stem+'.pt')).exists():
            tmp=out/(stem+'.pt.partial')
            with tmp.open('wb') as f:
                for block in volume.read_file('/'+trial+'/'+stem+'.pt'): f.write(block)
            tmp.replace(out/(stem+'.pt'))
            (out/'train.log').write_bytes(b''.join(volume.read_file('/'+trial+'/train.log')))
    (out/'report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(trial=trial,epoch=status['next_epoch'],complete=done,
                         accuracy=status['metrics']['valid/accuracy'],
                         margin=status['metrics']['gate/margin'],gate_failed=status['metrics']['gate/failed'])),flush=True)
    return done


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trial',required=True)
    p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        if collect(args.trial) or not args.watch: break
        time.sleep(30)

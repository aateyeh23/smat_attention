"""Collect the one-arm MQAR transport experiment and compare matched epochs."""
import argparse
import json
from pathlib import Path
import time
import modal

OUT=Path(__file__).resolve().parent/'results/mqar_gdn_transport/w32_d3_s123'
BASE=Path(__file__).resolve().parent/'results/mqar_width_sweeps/gdn'


def collect(neighbor_write_grad=False):
    volume=modal.Volume.from_name('smat-gdn-transport-w32-d3')
    out=OUT.with_name(OUT.name+'_neighbor_grad') if neighbor_write_grad else OUT
    remote='/seed123/neighbor_write_grad' if neighbor_write_grad else '/seed123'
    out.mkdir(parents=True,exist_ok=True)
    for name in ('validated.json','gpu-validation.log','w32-d3.json','w32-d3-history.jsonl'):
        try:
            raw=b''.join(volume.read_file(remote+'/'+name))
        except (FileNotFoundError,modal.exception.NotFoundError):
            continue
        (out/name).write_bytes(raw)
    status_path=out/'w32-d3.json'
    if not status_path.exists():
        print('Waiting for GPU validation / first epoch checkpoint',flush=True)
        return False
    status=json.loads(status_path.read_text())
    epoch=status['next_epoch']
    rows=[]
    series=[('GDN',BASE/'w32-d1-history.jsonl'),
                       ('Original SMAT d=3',BASE/'w32-d3-history.jsonl'),
                       ('Transport SMAT d=3',OUT/'w32-d3-history.jsonl')]
    if neighbor_write_grad:
        series.append(('Transport + neighbor write gradients',out/'w32-d3-history.jsonl'))
    for label,path in series:
        items=[json.loads(s) for s in path.read_text().splitlines()]
        item=next((x for x in items if x['epoch']==epoch),None)
        if item:
            rows.append(dict(model=label,epoch=epoch,**item['metrics']))
    report=['MQAR width 32, d=3: GDN boundary transport; one write and four reads.',
            '',f'Epoch {epoch}/32. Complete: {status["complete"]}. Seed 123, LR 0.01, two layers/heads, head/state 16.',
            '', '| Model at matched epoch | Overall accuracy | 64-pair accuracy | Validation loss |',
            '|---|---:|---:|---:|']
    for row in rows:
        report.append(f'| {row["model"]} | {100*row["valid/accuracy"]:.2f}% | {100*row["valid/num_kv_pairs/accuracy-64"]:.2f}% | {row["valid/loss"]:.3f} |')
    if neighbor_write_grad:
        report.extend(['','Neighbor-gradient run started fresh. Original transport paused at epoch 23;',
                       'its matched-epoch row is unavailable after epoch 23.'])
    (out/'report.md').write_text('\n'.join(report)+'\n')
    (out/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    if status['complete'] and not (out/'w32-d3.pt').exists():
        temporary=out/'w32-d3.pt.partial'
        with temporary.open('wb') as f:
            for chunk in volume.read_file(remote+'/w32-d3.pt'): f.write(chunk)
        temporary.replace(out/'w32-d3.pt')
        (out/'train.log').write_bytes(b''.join(volume.read_file(remote+'/train.log')))
    print(json.dumps(dict(epoch=epoch,complete=status['complete'],comparison=rows)),flush=True)
    return status['complete']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true')
    parser.add_argument('--neighbor-write-grad',action='store_true');args=parser.parse_args()
    while True:
        if collect(args.neighbor_write_grad) or not args.watch: break
        time.sleep(60)

"""Report every run and topology-averaged comparisons paired by training seed."""
import json
from pathlib import Path
import statistics
from common import ROOT, SEEDS, atomic_json


def main():
    tasks=json.loads((ROOT/'tasks.json').read_text())
    lines=['# Joint-recall incidence ablation','',
        'GDN, d=3, width 64, two layers; 32 epochs, LR 0.003. Three training seeds. '
        'Random incidence uses two graph seeds crossed with every training seed.','',
        'Geometry versus random changes only incidence. Independent-bucket controls change the reader; '
        'the byte-matched control also changes hash bin counts. Matrix-table bytes are not an implemented autoregressive cache.','',
        '| Run | Epoch | Status | Validation macro (%) | Final test macro (%) |',
        '|---|---:|---|---:|---:|']
    records=[];completed={}
    for task in tasks:
        path=Path(task['result']);folder=path.parent
        result=json.loads(path.read_text()) if path.exists() else None
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        metrics=folder/'metrics.jsonl'
        history=[json.loads(x) for x in metrics.read_text().splitlines()] if metrics.exists() else []
        validation=history[-1].get('accuracy') if history else None
        final=result['final_test']['accuracy'] if result else None
        fmt=lambda x: '—' if x is None else f'{100*x:.2f}'
        lines.append(f"| {task['name']} | {result['epochs'] if result else status.get('epoch',0)} | {'complete' if result else ('started' if status else 'pending')} | {fmt(validation)} | {fmt(final)} |")
        records.append(dict(task=task,status=status,result=result))
        if result:
            assert result['complete'] and result['epochs']==32 and result['steps']==22560
            completed[(task['variant'],task['topology'],task['seed'])]=result
    lines+=['','## Final test accuracy by load','','Only complete runs appear below.','',
            '| Run | 4 bindings | 16 | 128 | 256 | 512 |','|---|---:|---:|---:|---:|---:|']
    cells=['c1-k4','c2-k8','c8-k16','c16-k16','c32-k16']
    for task in tasks:
        result=completed.get((task['variant'],task['topology'],task['seed']))
        if result:
            scores=[result['final_test']['cells'][c]['accuracy']*100 for c in cells]
            lines.append('| '+task['name']+' | '+' | '.join(f'{x:.2f}' for x in scores)+' |')
    if len(completed)==len(tasks):
        lines+=['','## Geometry minus random incidence','','Random topology draws are averaged within each training seed. '
                'Reported variability is across the three paired training-seed differences.','',
                '| Cell | Mean difference (pp) | SD across training seeds (pp) |','|---|---:|---:|']
        for cell in cells:
            deltas=[]
            for seed in SEEDS:
                geo=completed['geometry',0,seed]['final_test']['cells'][cell]['accuracy']
                rand=statistics.mean(completed['random',t,seed]['final_test']['cells'][cell]['accuracy'] for t in (17,29))
                deltas.append(100*(geo-rand))
            lines.append(f'| {cell} | {statistics.mean(deltas):+.2f} | {statistics.stdev(deltas):.2f} |')
    atomic_json(ROOT/'summary.json',dict(completed=len(completed),total=len(tasks),runs=records))
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(completed)}/{len(tasks)} complete',flush=True)


if __name__=='__main__':
    main()

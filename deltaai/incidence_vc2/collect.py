import json
from pathlib import Path
from common import ROOT,BASE,atomic_json


def main():
    tasks=json.loads((ROOT/'tasks.json').read_text())
    records=[];done=0
    lines=['# Degree-matched VC-2 incidence control','',
        'Exact VC=2 applies to individual summary supports; four-read unions can have higher VC. '
        'Matrices come from a constrained random walk initialized from relabeled geometry, not uniform sampling.','',
        '| Run | Epoch | Complete | Latest validation macro (%) | Final test: 128 bindings (%) | Final test macro (%) |',
        '|---|---:|---|---:|---:|---:|']
    for task in tasks:
        path=Path(task['result']);folder=path.parent
        s=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        h=folder/'metrics.jsonl';history=[json.loads(l) for l in h.read_text().splitlines()] if h.exists() else []
        result=json.loads(path.read_text()) if path.exists() else None
        if result:assert result['complete'] and result['epochs']==32 and result['steps']==22560;done+=1
        fmt=lambda x:'—' if x is None else f'{100*x:.2f}'
        val=history[-1]['accuracy'] if history else None
        primary=result['final_test']['cells']['c8-k16']['accuracy'] if result else None
        macro=result['final_test']['accuracy'] if result else None
        lines.append(f"| {task['name']} | {s.get('epoch',0)} | {bool(result)} | {fmt(val)} | {fmt(primary)} | {fmt(macro)} |")
        records.append(dict(task=task,status=s,result=result))
    lines+=['','## Paired final-test comparison','','| Seed | Variant | 128 bindings (%) | Macro (%) |','|---:|---|---:|---:|']
    for seed in (123,456,789):
        for name,path in [('geometry',BASE/'geometry'),('random VC3, draw 17',BASE/'random-t17'),('random VC2',ROOT/'random_vc2-t17')]:
            p=path/f'shared-gdn_current-d3-w64-s{seed}/result.json'
            if p.exists():
                a=json.loads(p.read_text())['final_test']
                lines.append(f"| {seed} | {name} | {100*a['cells']['c8-k16']['accuracy']:.2f} | {100*a['accuracy']:.2f} |")
    atomic_json(ROOT/'summary.json',dict(completed=done,total=len(tasks),runs=records))
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'{done}/{len(tasks)} complete',flush=True)


if __name__=='__main__':main()

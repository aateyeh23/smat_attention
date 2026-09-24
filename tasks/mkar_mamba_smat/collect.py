"""Summarize independent fixed-k models, never selecting best seeds/checkpoints."""
import csv
import json
import statistics
from common import ROOT, CONFIGS, KS, atomic_json


def main():
    jobs=json.loads((ROOT/'tasks.json').read_text());rows=[]
    completed_jobs=0
    for job in jobs:
        completed_jobs+=int((ROOT/'runs'/job['name']/'result.json').exists())
        for k in KS:
            p=ROOT/'runs'/job['name']/f'k{k}'/'result.json'
            if p.exists():
                r=json.loads(p.read_text())
                rows.append(dict(config=job['config']['name'],seed=job['seed'],k=k,**r['evaluation']))
    if rows:
        with (ROOT/'results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    text=[f'# Exploratory Mamba writer-gradient pilot: {len(rows)}/3 fits; {completed_jobs}/1 jobs complete','',
        'Final-step exact-support accuracy (%), mean ± sample SD. Independent models for each k.','',
        '| Configuration | k=1 | k=2 | k=3 |','|---|---|---|---|']
    for config in CONFIGS:
        cells=[]
        for k in KS:
            group=[r['exact_support'] for r in rows if r['config']==config['name'] and r['k']==k]
            mean=statistics.mean(group) if group else None
            std=statistics.stdev(group) if len(group)>1 else None
            summary.append(dict(config=config['name'],k=k,seeds=len(group),mean=mean,std=std))
            cells.append('pending' if not group else f'{100*mean:.2f} ± '+
                ('—' if std is None else f'{100*std:.2f}')+f' (n={len(group)})')
        text.append('| '+config['name']+' | '+' | '.join(cells)+' |')
    (ROOT/'summary.md').write_text('\n'.join(text)+'\n')
    atomic_json(ROOT/'summary.json',dict(completed_fits=len(rows),completed_jobs=completed_jobs,summary=summary))
    print('COMPLETE',len(rows),'/3 fits;',completed_jobs,'/1 jobs',flush=True)


if __name__=='__main__':main()

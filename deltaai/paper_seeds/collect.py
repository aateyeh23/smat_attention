"""Aggregate final-epoch measurements; retain per-seed and per-cell results."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent/'results/paper_seeds_20260920'
planned = json.loads((ROOT/'tasks.json').read_text())
original = json.loads((ROOT/'original_runs.json').read_text())
rows = []
for task in original+planned:
    path = Path(task['result'])
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    assert result['complete'], path
    common = {k:task[k] for k in ['task','family','width','d','seed']}
    if task['task'] == 'mqar':
        assert result['next_epoch'] == 32, path
        metrics = result['metrics']
        rows.append(dict(common, split='validation', cell='macro', accuracy=100*metrics['valid/accuracy'], source=str(path)))
        for key, value in metrics.items():
            if key.startswith('valid/num_kv_pairs/accuracy-'):
                rows.append(dict(common, split='validation', cell=key.rsplit('-',1)[1], accuracy=100*value, source=str(path)))
    else:
        assert result['epochs'] == 32 and result['steps'] == 22560, path
        history=[json.loads(line) for line in (path.parent/'metrics.jsonl').read_text().splitlines()]
        final,=[row for row in history if row['epoch']==32]
        for split, metrics in [('validation',final),('test',result['final_test'])]:
            rows.append(dict(common, split=split, cell='macro', accuracy=100*metrics['accuracy'],source=str(path)))
            for cell, metric in metrics['cells'].items():
                rows.append(dict(common,split=split,cell=cell,accuracy=100*metric['accuracy'],source=str(path)))
groups={}
for row in rows:
    key=tuple(row[k] for k in ['task','family','width','d','split','cell'])
    groups.setdefault(key,[]).append(row)
summary=[]
for key, group in sorted(groups.items()):
    seeds=[r['seed'] for r in group];assert len(seeds)==len(set(seeds)),key
    values=[r['accuracy'] for r in group]
    summary.append(dict(zip(['task','family','width','d','split','cell'],key),
                        n=len(values),seeds=seeds,mean=statistics.mean(values),
                        std=statistics.stdev(values) if len(values)>1 else None))
for filename, data in [('per_seed.csv',rows),('summary.csv',summary)]:
    with (ROOT/filename).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]) if data else [])
        writer.writeheader();writer.writerows(data)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
count=sum(Path(task['result']).exists() for task in planned)
lines=[f'# Paper seed campaign: {count}/{len(planned)} new runs complete','',
       'Final epoch only. Values are percent accuracy; SD is the sample standard deviation across training seeds.',
       'MQAR data and whole-batch ordering remain fixed; joint-recall data remain fixed and batch ordering follows the original seed-dependent trainer.',
       'Table 3 uses equal-cell validation accuracy. Independent final-test results are retained separately.', '',
       '| Task | Backbone | Width | Variant | Seeds completed | Mean | SD |',
       '|---|---|---:|---|---:|---:|---:|']
for r in summary:
    if r['split']!='validation' or r['cell']!='macro':continue
    variant='Log-Linear' if r['d']==0 else ('Native' if r['d']==1 else f"SMat d={r['d']}")
    sd='—' if r['std'] is None else f"{r['std']:.2f}"
    lines.append(f"| {r['task']} | {r['family']} | {r['width']} | {variant} | {r['n']}/5 | {r['mean']:.2f} | {sd} |")
lines += ['', 'The draft’s GDN width-64 d=2 entry is 89.71; the saved seed-123 final metric is 89.6834375. Aggregation uses the saved metric.',
          'Original Table 2 Log-Linear results are verified from SCF commit 3265a5df. Unfinished runs are never counted as final results.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
lookup={(r['task'],r['family'],r['width'],r['d']):r for r in summary
        if r['split']=='validation' and r['cell']=='macro'}
def tex_cell(task,family,width,d):
    r=lookup.get((task,family,width,d))
    if r is None or r['n']!=5:return r'\text{pending}'
    return f"{r['mean']:.2f} \\pm {r['std']:.2f}"
table2=[r'\begin{tabular}{llccccc}',r'\toprule',
        r'Backbone & Width & Native & Log-Linear & $d=2$ & $d=3$ & $d=4$ \\',r'\midrule']
for family,label in [('gdn','Gated DeltaNet'),('mamba2','Mamba-2')]:
    for width in [16,32,64]:
        table2.append(f'{label} & {width} & '+' & '.join('$'+tex_cell('mqar',family,width,d)+'$' for d in [1,0,2,3,4])+r' \\')
table2 += [r'\bottomrule',r'\end{tabular}']
table3=[r'\begin{tabular}{lcc}',r'\toprule',r'Variant & Mamba-2 & GDN \\',r'\midrule']
for d,label in [(1,'Native'),(0,'Log-Linear'),(2,r'+ SMat ($d=2$)'),(3,r'+ SMat ($d=3$)'),(4,r'+ SMat ($d=4$)')]:
    task='joint_loglinear' if d==0 else 'joint'
    families=['mamba2','gdn' if d==0 else 'gdn_current']
    table3.append(label+' & '+' & '.join('$'+tex_cell(task,f,64,d)+'$' for f in families)+r' \\')
table3 += [r'\bottomrule',r'\end{tabular}']
for filename,lines in [('table2.tex',table2),('table3.tex',table3)]:
    (ROOT/filename).write_text('\n'.join(lines)+'\n')
print(f'{count}/{len(planned)} new runs complete')

"""Progress and final per-load tables for the fixed-dataset joint-recall study."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joint_recall import ROOT,CELLS

rows=[];progress=[];histories=[]
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir() or not (folder/'recipe.json').exists():continue
    q=json.loads((folder/'recipe.json').read_text())
    status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    history=[]
    if (folder/'metrics.jsonl').exists():
        for line in (folder/'metrics.jsonl').read_text().splitlines():
            try:history.append(json.loads(line))
            except json.JSONDecodeError:pass
    histories.append((q,history))
    completed=(folder/'result.json').exists()
    progress.append(dict(run=folder.name,complete=completed,epoch=status.get('epoch',0),steps=status.get('steps',0),
                         validation_accuracy=history[-1]['accuracy'] if history else None))
    if completed:
        r=json.loads((folder/'result.json').read_text())
        for endpoint in ['final_test','best_test']:
            for cell,m in r[endpoint]['cells'].items():
                rows.append(dict(condition=q['condition'],family=q['family'],d=q['d'],width=q['width'],seed=q['seed'],
                    endpoint=endpoint,cell=cell,epoch=q['epochs'] if endpoint=='final_test' else r['best_validation_epoch'],
                    steps=r['steps'],parameters=r['parameters'],trainable_parameters=r['trainable_parameters'],**m))
with (ROOT/'progress.csv').open('w') as f:
    w=csv.DictWriter(f,['run','complete','epoch','steps','validation_accuracy']);w.writeheader();w.writerows(progress)
if rows:
    with (ROOT/'results.csv').open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def fmt(v):return f'{100*st.mean(v):.2f}'+(f' ± {100*st.stdev(v):.2f}' if len(v)>1 else '')
labels={'mamba2':'Mamba-2','gdn_current':'GDN'}
report=['# Block-context joint recall: full epoch budget','',
        f'Completed {sum(p["complete"] for p in progress)} of 24 planned runs. '
        'Each run:180k fixed training examples, batch256,32 epochs,22,624 updates. '
        'Incomplete runs are progress only, not final results. Mean ± sample SD across available seeds.','',
        '| Run | Complete | Epoch | Updates | Latest validation accuracy (%) |','|---|---|---:|---:|---:|']
for p in progress:
    val='—' if p['validation_accuracy'] is None else f'{100*p["validation_accuracy"]:.2f}'
    report.append(f'| {p["run"]} | {p["complete"]} | {p["epoch"]} | {p["steps"]} | {val} |')
for endpoint in ['final_test','best_test']:
    report += ['',f'## {endpoint}','',
      '| Condition | Model | Table | Seeds | Accuracy (%) | All-query exact (%) |','|---|---|---|---:|---:|---:|']
    groups=defaultdict(list)
    for r in rows:
        if r['endpoint']==endpoint:groups[(r['condition'],r['family'],r['d'],r['cell'])].append(r)
    for (condition,family,d,cell),rr in sorted(groups.items()):
        label=labels[family]+(f' + SMat d{d}' if d>1 else '')
        report.append(f'| {condition} | {label} | {cell} | {len(rr)} | {fmt([r["accuracy"] for r in rr])} | {fmt([r["exact_accuracy"] for r in rr])} |')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
colors={('mamba2',1):'#D55E00',('mamba2',3):'#E69F00',('gdn_current',1):'#0072B2',('gdn_current',3):'#009E73'}
for ax,condition in zip(axes,['shared','unique']):
    labeled=set()
    for q,history in histories:
        if q['condition']!=condition or not history:continue
        key=(q['family'],q['d'])
        label=labels[key[0]]+(' baseline' if key[1]==1 else ' + SMat d3')
        ax.plot([r['epoch'] for r in history],[100*r['accuracy'] for r in history],color=colors[key],alpha=.7,
                linestyle={123:'-',456:'--',789:':'}[q['seed']],label=label if key not in labeled else None)
        labeled.add(key)
    ax.set(title=condition+' keys',xlabel='Completed epochs',ylabel='Validation macro accuracy (%)',xlim=(0,32),ylim=(0,101))
    ax.axhline(6.25,color='gray',ls=':',lw=1);ax.grid(alpha=.2)
    if labeled:ax.legend(fontsize=8)
fig.suptitle('Joint recall: validation learning curves; solid/dashed/dotted = seeds123/456/789')
fig.tight_layout();fig.savefig(ROOT/'learning_curves.png',dpi=170);fig.savefig(ROOT/'learning_curves.pdf');plt.close(fig)
report += ['','![Validation learning curves](learning_curves.png)','']
(ROOT/'report.md').write_text('\n'.join(report)+'\n')
print(f'{sum(p["complete"] for p in progress)}/24 complete; {len(progress)} runs started')

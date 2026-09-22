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
            metrics=dict(r[endpoint]['cells'])
            metrics['macro']={key:sum(m[key] for m in metrics.values())/len(metrics)
                              for key in next(iter(metrics.values()))}
            for cell,m in metrics.items():
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
        'Incomplete runs are progress only, not final results. Mean ± sample SD across available seeds. '
        'Macro scores weight the five table sizes equally.','',
        '| Run | Complete | Epoch | Updates | Latest validation accuracy (%) |','|---|---|---:|---:|---:|']
for p in progress:
    val='—' if p['validation_accuracy'] is None else f'{100*p["validation_accuracy"]:.2f}'
    report.append(f'| {p["run"]} | {p["complete"]} | {p["epoch"]} | {p["steps"]} | {val} |')
if (ROOT/'lookup_controls.json').exists():
    report += ['','## Analytical lookup controls','',
        'These controls read the correct information table but discard context or key identity. '
        'They are evaluated on all test queries without fitting any parameters. '
        'Uniform guessing among value tokens scores 6.25%; context-blind lookup can score much higher.','',
        '| Condition | Table | Key-only majority (%) | Last value for key (%) | Context-only majority (%) |',
        '|---|---|---:|---:|---:|']
    for r in json.loads((ROOT/'lookup_controls.json').read_text()):
        report.append(f'| {r["condition"]} | {r["cell"]} | {100*r["key_only_majority_accuracy"]:.2f} | '
                      f'{100*r["last_value_for_key_accuracy"]:.2f} | {100*r["context_only_majority_accuracy"]:.2f} |')
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
fig,axes=plt.subplots(2,5,figsize=(17,6.5),sharex=True,sharey=True)
for ri,condition in enumerate(['shared','unique']):
    for ci,(c,k,_) in enumerate(CELLS):
        ax=axes[ri,ci];cell=f'c{c}-k{k}'
        for q,history in histories:
            if q['condition']!=condition or not history:continue
            key=(q['family'],q['d'])
            ax.plot([r['epoch'] for r in history],[100*r['cells'][cell]['accuracy'] for r in history],
                    color=colors[key],alpha=.7,linestyle={123:'-',456:'--',789:':'}[q['seed']])
        ax.set(title=f'{condition}: C={c}, K={k}',xlim=(0,32),ylim=(0,101))
        ax.grid(alpha=.2)
        if ri==1:ax.set_xlabel('Completed epochs')
        if ci==0:ax.set_ylabel('Validation accuracy (%)')
fig.suptitle('Joint recall by table size; same model colors and seed styles as macro curves')
fig.tight_layout();fig.savefig(ROOT/'learning_curves_by_load.png',dpi=170);fig.savefig(ROOT/'learning_curves_by_load.pdf');plt.close(fig)
report += ['','![Validation learning curves](learning_curves.png)',
           '','![Validation learning curves by table size](learning_curves_by_load.png)','']
(ROOT/'report.md').write_text('\n'.join(report)+'\n')
finished=sum(p['complete'] for p in progress)
findings=['# Joint recall findings','',f'Completed {finished}/24 planned runs.','']
if finished<24:
    findings += ['The campaign is still running. Available results and curves are interim; '
                 'do not treat incomplete seed averages as the final experiment.','']
findings += ['Each completed run used180,000 fixed training examples for32 epochs '
             '(22,624 optimizer updates;5.76M example presentations). '
             'Final-epoch test is the primary endpoint.','',
             '| Condition | Backbone | Paired seeds | Native macro test (%) | SMat macro test (%) | SMat − native (pp) |',
             '|---|---|---:|---:|---:|---:|']
for condition in ['shared','unique']:
    for family in ['mamba2','gdn_current']:
        selected={(r['d'],r['seed']):r for r in rows if r['condition']==condition and r['family']==family
                  and r['endpoint']=='final_test' and r['cell']=='macro'}
        seeds=[s for s in [123,456,789] if (1,s) in selected and (3,s) in selected]
        if not seeds:continue
        native=[selected[1,s]['accuracy'] for s in seeds];smat=[selected[3,s]['accuracy'] for s in seeds]
        delta=[b-a for a,b in zip(native,smat)]
        findings.append(f'| {condition} | {labels[family]} | {len(seeds)} | {fmt(native)} | {fmt(smat)} | {fmt(delta)} |')
findings += ['','Mean ± sample SD across paired seeds; three seeds do not establish statistical significance. '
             'A positive difference favors SMat at this fixed data-exposure budget. '
             'Models are not parameter-, compute-, or memory-matched.','',
             'Inspect the per-table results and analytical lookup controls in [report.md](report.md). '
             'Scoring above random guessing alone does not establish context binding: '
             'a context-blind lookup can recover many targets. '
             'The paired unique-key condition measures how much performance changes when context is needed.','',
             'This is an adaptation of published block-context joint recall: masked inquiry answers, '
             'smaller table-size grid and the MQAR training budget. '
             'The equal information/inquiry boundary aligns with the SMat reset. '
             'Splits are independent draws from the same distribution; this is not a held-out-combination test. '
             'See [PROTOCOL.md](PROTOCOL.md) for the complete design and failed unpadded pilot.','']
(ROOT/'FINDINGS.md').write_text('\n'.join(findings)+'\n')
print(f'{sum(p["complete"] for p in progress)}/24 complete; {len(progress)} runs started')

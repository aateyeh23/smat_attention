"""Render every locked SMat dimension on the same fresh confirmation tables."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args()
r=json.loads((args.folder/'result.json').read_text());assert r['complete']
names={'mamba2':'Mamba-2','gdn_current':'Gated DeltaNet'}
ds=sorted({m['d'] for m in r['selection']['models']} - {1})
colors={2:'#0072B2',3:'#D55E00',4:'#009E73'}
fig,axes=plt.subplots(2,2,figsize=(12,8),gridspec_kw={'height_ratios':[2,1]})
report=['# Fresh-table joint-recall confirmation','',
        f'Independent split code {r["split_code"]}; {r["examples_per_cell"]:,} tables per cell. '
        'All final checkpoints were locked before evaluating these tables. '
        'All models trained on the same 180,000-example dataset for 32 epochs with seed 123. '
        'The primary dimension sweep uses LR0.003; native GDN LR0.01 is an additional control.','',
        'Intervals are 95% paired table-bootstrap intervals, conditional on this training seed. '
        'They do not quantify variation across training seeds. The prespecified primary '
        'positive-result criterion concerns d=3 in both backbones; the other dimensions '
        'are reported in full, with no selection based on fresh-table performance.','',
        '| Backbone | d | Cell | Native (%) | SMat (%) | Difference (pp), 95% CI | Single-field oracle (%) |',
        '|---|---:|---|---:|---:|---|---:|']
for col,family in enumerate(names):
    for index,d in enumerate(ds):
        rows=[a for a in r['comparisons'] if a['family']==family and a.get('d',3)==d]
        x=np.arange(len(rows));native=np.array([a['native_accuracy'] for a in rows])*100
        smat=np.array([a['smat_accuracy'] for a in rows])*100
        control=np.array([a['best_single_field_accuracy'] for a in rows])*100
        delta=np.array([a['smat_minus_native']['difference'] for a in rows])*100
        ci=np.array([a['smat_minus_native']['ci95'] for a in rows])*100
        for a in rows:
            interval=a['smat_minus_native'];lo,hi=interval['ci95']
            report.append(f'| {names[family]} | {d} | {a["cell"]} | {a["native_accuracy"]*100:.2f} | '
                          f'{a["smat_accuracy"]*100:.2f} | {interval["difference"]*100:+.2f} '
                          f'[{lo*100:+.2f}, {hi*100:+.2f}] | {a["best_single_field_accuracy"]*100:.2f} |')
        ax=axes[0,col]
        if index==0:
            ax.plot(x,native,'o-',color='#333333',label='Native')
            ax.plot(x,control,'--',color='#888888',label='Best single-field oracle')
            ax.axhline(6.25,color='#bbbbbb',ls=':',lw=1)
        chosen=next(m for m in r['selection']['models'] if m['family']==family and m['d']==d)
        recipe=json.loads((Path(chosen['folder'])/'recipe.json').read_text())
        label=f'SMat d={d}'+(' (no scalar decay)' if recipe.get('gdn_memory_scalar_decay') is False else '')
        ax.plot(x,smat,'o-',color=colors[d],label=label)
        ax=axes[1,col];width=.75/len(ds);offset=(index-(len(ds)-1)/2)*width
        ax.bar(x+offset,delta,color=colors[d],width=width,label=f'd={d}')
        ax.errorbar(x+offset,delta,yerr=np.maximum(0,np.stack((delta-ci[:,0],ci[:,1]-delta))),
                    fmt='none',ecolor='black',capsize=3)
    for chosen in r['selection'].get('additional_models',[]):
        if chosen['family']!=family:continue
        q=json.loads((Path(chosen['folder'])/'recipe.json').read_text())
        extra=r['metrics'][chosen['label']]['cells']
        axes[0,col].plot(x,[extra[a['cell']]['accuracy']*100 for a in rows],
                         's--',color='#666666',label=f'Native LR={q["lr"]}')
    axes[0,col].set(title=names[family],ylabel='Query accuracy (%)',ylim=(0,103))
    axes[0,col].legend(fontsize=8)
    axes[1,col].axhline(0,color='black',lw=.8)
    axes[1,col].set(ylabel='SMat − native (pp)',xlabel='Bindings (contexts × keys)')
    labels=[]
    for a in rows:
        c,k=[int(part[1:]) for part in a['cell'].split('-')];labels.append(f'{c*k}\n({c}×{k})')
    for ax in axes[:,col]:ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2)
fig.suptitle('Joint context–key recall · seed 123 · fresh confirmation tables')
fig.tight_layout();fig.savefig(args.folder/'confirmation.png',dpi=200)
fig.savefig(args.folder/'confirmation.pdf');plt.close(fig)
report+=['','## Equal-cell macro accuracy','',
         '| Backbone | d | Native (%) | SMat (%) | Difference (pp), 95% CI |',
         '|---|---:|---:|---:|---|']
for a in r['macro_comparisons']:
    interval=a['smat_minus_native'];lo,hi=interval['ci95']
    report.append(f'| {names[a["family"]]} | {a.get("d",3)} | {a["native_accuracy"]*100:.2f} | '
                  f'{a["smat_accuracy"]*100:.2f} | {interval["difference"]*100:+.2f} '
                  f'[{lo*100:+.2f}, {hi*100:+.2f}] |')
if r.get('additional_comparisons'):
    report+=['','## Additional native controls','',
             'A native setting can be strongest at high load while another is strongest in macro accuracy. '
             'Both are included when this happens; no high-load advantage is claimed against only the weaker control.',
             '','| Native setting | SMat d | Cell | Native (%) | SMat (%) | Difference (pp), 95% CI |',
             '|---|---:|---|---:|---:|---|']
    for a in r['additional_comparisons']+r['additional_macro_comparisons']:
        interval=a['smat_minus_native'];lo,hi=interval['ci95']
        report.append(f'| {a["native_label"]} | {a["d"]} | {a.get("cell","macro")} | '
                      f'{a["native_accuracy"]*100:.2f} | {a["smat_accuracy"]*100:.2f} | '
                      f'{interval["difference"]*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] |')
report+=['','## Selected training settings','',
         '| Model | Development trial | LR | Weight decay | Parameters | Trainable parameters | GDN extra-memory scalar decay |',
         '|---|---|---:|---:|---:|---:|---|']
for m in r['selection']['models']+r['selection'].get('additional_models',[]):
    folder=Path(m['folder']);q=json.loads((folder/'recipe.json').read_text())
    trained=json.loads((folder/'result.json').read_text())
    label=names[m['family']]+(' native' if m['d']==1 else f' + SMat d={m["d"]}')
    decay=str(q.get('gdn_memory_scalar_decay',True)) if m['family']=='gdn_current' and m['d']>=2 else 'N/A'
    report.append(f'| {label} | {folder.parent.name} | {q["lr"]} | {q["weight_decay"]} | '
                  f'{trained["parameters"]:,} | {trained["trainable_parameters"]:,} | {decay} |')
report+=['','Backbone width and depth are matched. Total parameter counts are not matched; '
         'these results do not isolate the benefit of the mask from the additional parameters.',
         '','## Exact-table accuracy','',
         '| Model | Macro (%) | 4 bindings (%) | 16 (%) | 128 (%) | 256 (%) | 512 (%) |',
         '|---|---:|---:|---:|---:|---:|---:|']
cell_names=list(r['metrics']['mamba2-d1']['cells'])
for chosen in r['selection']['models']+r['selection'].get('additional_models',[]):
    label=chosen.get('label',f'{chosen["family"]}-d{chosen["d"]}')
    metric=r['metrics'][label]
    values=[metric['exact_accuracy']]+[metric['cells'][cell]['exact_accuracy'] for cell in cell_names]
    report.append('| '+label+' | '+' | '.join(f'{value*100:.2f}' for value in values)+' |')
report+=['','## Single-field controls','',
         '| Cell | Key-only majority (%) | Context-only majority (%) | Better rule per table (%) |',
         '|---|---:|---:|---:|']
for a in r['comparisons']:
    if a['family']!='mamba2' or a['d']!=ds[0]:continue
    report.append(f'| {a["cell"]} | {a["key_only_majority_accuracy"]*100:.2f} | '
                  f'{a["context_only_majority_accuracy"]*100:.2f} | '
                  f'{a["best_single_field_accuracy"]*100:.2f} |')
report+=['','The complete development history, including failed trials, is retained in the parent report. '
         'These splits sample fresh tables from the same identifier distribution; this is not a held-out-combination claim.',
         '','The single-field control chooses, for each table, the better of the key-only '
         'and context-only majority predictors. Both know all stored values but ignore one identifier. '
         'The saved result and oracle arrays retain the two controls separately.',
         '','![Fresh-table results](confirmation.png)','']
(args.folder/'report.md').write_text('\n'.join(report)+'\n')
print(args.folder/'report.md')

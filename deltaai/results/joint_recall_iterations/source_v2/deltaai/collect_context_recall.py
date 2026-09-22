"""Summarize completed context-binding experiments and their learning curves."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent/'results/context_recall'
rows, histories = [], []
for p in sorted(root.glob('*/result.json')):
    recipe = json.loads((p.parent/'recipe.json').read_text())
    rows.append(dict(recipe, **json.loads(p.read_text())))
for p in sorted(root.glob('*/metrics.jsonl')):
    recipe = json.loads((p.parent/'recipe.json').read_text())
    histories.append((recipe, [json.loads(s) for s in p.read_text().splitlines()]))
if rows:
    with (root/'results.csv').open('w') as f:
        w = csv.DictWriter(f, sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader(); w.writerows(rows)
groups = defaultdict(list)
for r in rows: groups[(r['width'],r['load'],r['contexts'],r['min_length'])].append(r)
report = ['# Context-dependent recall at 500 steps', '',
          'Explicit-context triples; fixed record count and length within each reuse comparison. '
          'Mean ± sample SD when multiple seeds are available. Full-vocabulary predictions, '
          '16 equiprobable answer values (6.25% value chance). Context gap compares a prediction '
          'with the queried context versus another context for the same key. It is undefined '
          'when keys are unique. No claim of an exact replication of the block-context paper task.', '']
colors = {1:'#222222',2:'#0072B2',3:'#D55E00',4:'#009E73'}
labels = {'mamba2':'Mamba-2','gdn_current':'GDN'}
for (width,load,contexts,length), runs in groups.items():
    name = f'w{width}-r{load}-c{contexts}-t{length}'
    report += [f'## {load} records, {contexts} contexts, width {width}, length {length}', '',
               '| Model | Shared fraction | Seeds | Accuracy (%) | Exact four-query (%) | Context gap (pp) | Gain vs backbone (pp) | Total / trainable parameters |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    cells = defaultdict(list)
    lookup = {(r['family'],r['d'],r['shared'],r['seed']):r for r in runs}
    for r in runs: cells[(r['family'],r['d'],r['shared'])].append(r)
    def fmt(v):
        return f'{st.mean(v):.2f}' + (f' ± {st.stdev(v):.2f}' if len(v)>1 else '')
    for (family,d,shared), rr in sorted(cells.items()):
        label=labels[family]+(f' + SMat d={d}' if d>1 else '')
        acc=fmt([100*r['accuracy'] for r in rr]); exact=fmt([100*r['exact_accuracy'] for r in rr])
        gap=fmt([r['context_binding_gap_pp'] for r in rr]) if shared>0 else '—'
        bases=[lookup.get((family,1,shared,r['seed'])) for r in rr]
        gain=fmt([100*(r['accuracy']-b['accuracy']) for r,b in zip(rr,bases)]) if d>1 and all(bases) else '—'
        report.append(f'| {label} | {shared:g} | {len(rr)} | {acc} | {exact} | {gap} | {gain} | {rr[0]["parameters"]:,} / {rr[0]["trainable_parameters"]:,} |')
    report += ['', '| Model | Paired seeds | Gain with unique keys (pp) | Gain with shared keys (pp) | Change in advantage (pp) |',
               '|---|---:|---:|---:|---:|']
    for family in ['mamba2','gdn_current']:
        for d in [2,3,4]:
            seeds=sorted({r['seed'] for r in runs if r['family']==family and r['d']==d})
            seeds=[s for s in seeds if all((family,dd,reuse,s) in lookup for dd in [1,d] for reuse in [0.,1.])]
            if not seeds: continue
            def differences(reuse):
                return [100*(lookup[(family,d,reuse,s)]['accuracy']-lookup[(family,1,reuse,s)]['accuracy']) for s in seeds]
            unique, shared = differences(0.), differences(1.)
            report.append(f'| {labels[family]} + SMat d={d} | {len(seeds)} | {fmt(unique)} | {fmt(shared)} | {fmt([s-u for s,u in zip(shared,unique)])} |')
    fig, axes = plt.subplots(2,2,figsize=(10,7),squeeze=False)
    for j,family in enumerate(['mamba2','gdn_current']):
        for d in [1,2,3,4]:
            xx=sorted(s for f,dd,s in cells if f==family and dd==d)
            if not xx: continue
            vv=[[100*r['accuracy'] for r in cells[(family,d,s)]] for s in xx]
            axes[0,j].errorbar(xx,[st.mean(v) for v in vv],yerr=[st.stdev(v) if len(v)>1 else 0 for v in vv],marker='o',capsize=3,color=colors[d],label='Baseline' if d==1 else f'SMat d={d}')
            bystep=defaultdict(list)
            for recipe,history in histories:
                if (recipe['width'],recipe['load'],recipe['contexts'],recipe['min_length'],recipe['shared'],recipe['family'],recipe['d']) != (width,load,contexts,length,1.,family,d): continue
                for row in history:bystep[row['step']].append(100*row['accuracy'])
            steps=sorted(bystep)
            if steps:
                means=[st.mean(bystep[s]) for s in steps]
                stds=[st.stdev(bystep[s]) if len(bystep[s])>1 else 0 for s in steps]
                axes[1,j].plot(steps,means,color=colors[d],label='Baseline' if d==1 else f'SMat d={d}')
                axes[1,j].fill_between(steps,[m-s for m,s in zip(means,stds)],[m+s for m,s in zip(means,stds)],color=colors[d],alpha=.1)
        shared_values=sorted({r['shared'] for r in runs})
        controls=[]
        for s in shared_values:
            byseed={r['seed']:100*r['key_only_oracle_accuracy'] for r in runs if r['shared']==s}
            controls.append(st.mean(byseed.values()))
        axes[0,j].plot(shared_values,controls,'--',color='gray',label='Ignores-context oracle')
        axes[0,j].set(title=labels[family],xlabel='Fraction of keys shared across contexts',ylabel='Test accuracy (%)',ylim=(0,101),xticks=shared_values)
        axes[1,j].set(title='Fully shared keys: learning curve',xlabel='Training steps',ylabel='Validation accuracy (%)',ylim=(0,101))
        for ax in axes[:,j]:ax.axhline(6.25,color='gray',ls=':',lw=1);ax.grid(alpha=.2);ax.legend(fontsize=8)
    counts=[len(rr) for rr in cells.values()]
    seed_label=str(min(counts)) if min(counts)==max(counts) else f'{min(counts)}–{max(counts)}'
    fig.suptitle(f'Context recall: {load} records, {contexts} contexts, T={length}, width={width}, 500 steps\nn={seed_label} seeds; bars/bands show sample SD where available')
    fig.tight_layout();fig.savefig(root/f'{name}.png',dpi=170);fig.savefig(root/f'{name}.pdf');plt.close(fig)
    report += ['',f'![{name}]({name}.png)','']
(root/'report.md').write_text('\n'.join(report)+'\n')
print(f'Collected {len(rows)} completed context-recall runs.')

"""Summarize matched component controls and all truth-table cases."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent/'results/boolean_recall'
rows=[dict(json.loads((p.parent/'recipe.json').read_text()),**json.loads(p.read_text()))
      for p in sorted(root.glob('*/result.json'))]
if not rows:raise SystemExit('No completed runs')
with (root/'results.csv').open('w') as fp:
    w=csv.DictWriter(fp,sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
def fmt(v):
    return f'{100*st.mean(v):.2f}'+(f' ± {100*st.stdev(v):.2f}' if len(v)>1 else '')
labels={'mamba2':'Mamba-2','gdn_current':'GDN'}
conditions=['direct_and','retrieval','retrieval_and']
titles={'direct_and':'Supplied bits → AND','retrieval':'Two keys → two bits','retrieval_and':'Two keys → AND'}
groups=defaultdict(list)
for r in rows:groups[(r['width'],r['load'],r['sequence_length'])].append(r)
report=['# Retrieval and conjunction at 500 steps','',
        'Mean ± sample SD when multiple seeds are available. Accuracy values are percentages. '
        'Retrieval exact accuracy requires both bits correct; AND exact accuracy requires its one answer correct. '
        'Every test has equal00/01/10/11 counts. Always-zero AND accuracy is75%, so inspect all four cases.','']
for (width,load,length),runs in sorted(groups.items()):
    report += [f'## {load} records, width{width}, T{length}','']
    cells=defaultdict(list)
    for r in runs:cells[(r['condition'],r['family'],r['d'])].append(r)
    for condition in conditions:
        report += [f'### {titles[condition]}','',
          '| Model | Seeds | Accuracy | Exact | 00 exact | 01 exact | 10 exact | 11 exact | Worst-case exact |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for (c,f,d),rr in sorted(cells.items()):
            if c!=condition:continue
            label=labels[f]+(f' + SMat d{d}' if d>1 else '')
            keys=['accuracy','exact_accuracy']+[f'truth_{case:02b}_exact_accuracy' for case in range(4)]+['worst_truth_exact_accuracy']
            report.append(f'| {label} | {len(rr)} | '+' | '.join(fmt([r[k] for r in rr]) for k in keys)+' |')
        report += ['']
        count_controls={r['seed']:r['query_blind_count_oracle_accuracy'] for r in runs if r['condition']==condition}
        if count_controls:
            report += [f'Query-blind memory-count oracle, per-target accuracy: {fmt(list(count_controls.values()))}%. '
                       'This oracle knows the memory bit counts but ignores the queried keys.','']
    report += ['| Model | Seeds | AND composed from retrieved bits | End-to-end AND | End-to-end AND balanced accuracy |',
               '|---|---:|---:|---:|---:|']
    for f,d in sorted({(r['family'],r['d']) for r in runs}):
        rr=cells.get(('retrieval',f,d),[]);aa=cells.get(('retrieval_and',f,d),[])
        if not rr or not aa:continue
        common={r['seed'] for r in rr}&{r['seed'] for r in aa}
        rr=[r for r in rr if r['seed'] in common];aa=[r for r in aa if r['seed'] in common]
        label=labels[f]+(f' + SMat d{d}' if d>1 else '')
        report.append(f'| {label} | {len(common)} | {fmt([r["composed_and_accuracy"] for r in rr])} | {fmt([r["accuracy"] for r in aa])} | {fmt([r["balanced_accuracy"] for r in aa])} |')
    report += ['', '| Model | Paired seeds | AND accuracy gain over backbone (pp) | Total / trainable parameters |',
               '|---|---:|---:|---:|']
    lookup={(r['condition'],r['family'],r['d'],r['seed']):r for r in runs}
    for f,d in sorted({(r['family'],r['d']) for r in runs}):
        rr=cells.get(('retrieval_and',f,d),[])
        if not rr:continue
        bases=[lookup.get(('retrieval_and',f,1,r['seed'])) for r in rr]
        gain=fmt([r['accuracy']-b['accuracy'] for r,b in zip(rr,bases)]) if d>1 and all(bases) else '—'
        label=labels[f]+(f' + SMat d{d}' if d>1 else '')
        report.append(f'| {label} | {len(rr)} | {gain} | {rr[0]["parameters"]:,} / {rr[0]["trainable_parameters"]:,} |')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,c in zip(axes,conditions):
        cc=sorted((f,d,rr) for (condition,f,d),rr in cells.items() if condition==c)
        means=[100*st.mean(r['exact_accuracy'] for r in rr) for _,_,rr in cc]
        stds=[100*st.stdev(r['exact_accuracy'] for r in rr) if len(rr)>1 else 0 for _,_,rr in cc]
        ax.bar(range(len(cc)),means,yerr=stds,capsize=3,color=['#0072B2' if f=='gdn_current' else '#D55E00' for f,d,rr in cc])
        ax.set(title=titles[c],ylim=(0,103),ylabel='Exact accuracy (%)',xticks=range(len(cc)),
               xticklabels=[labels[f]+('\nbase' if d==1 else f'\nd{d}') for f,d,rr in cc])
        ax.axhline(25 if c=='retrieval' else 75,color='gray',ls='--',label='Constant zero output')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    counts=[len(rr) for rr in cells.values()]
    n=str(min(counts)) if min(counts)==max(counts) else f'{min(counts)}–{max(counts)}'
    fig.suptitle(f'{load} records, T{length}, width{width}, 500 steps; n={n} seeds\nSample SD shown when n>1; inspect truth-case tables for shortcuts')
    fig.tight_layout();name=f'w{width}-r{load}-t{length}'
    fig.savefig(root/f'{name}.png',dpi=170);fig.savefig(root/f'{name}.pdf');plt.close(fig)
    report += ['',f'![{name}]({name}.png)','']
(root/'report.md').write_text('\n'.join(report)+'\n')
print(f'Collected {len(rows)} completed runs')

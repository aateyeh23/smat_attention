"""Collect database-set selection results without hiding class imbalance."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent/'results/database_recall'
rows = [dict(json.loads((p.parent/'recipe.json').read_text()), **json.loads(p.read_text()))
        for p in sorted(root.glob('*/result.json'))]
if not rows: raise SystemExit('No completed runs')
with (root/'results.csv').open('w') as fp:
    w = csv.DictWriter(fp, sorted(set().union(*(r.keys() for r in rows))))
    w.writeheader(); w.writerows(rows)
groups = defaultdict(list)
for r in rows: groups[(r.get('loss_mode','unweighted'), r['width'], r['load'], r['sequence_length'])].append(r)
labels = {'mamba2':'Mamba-2', 'gdn_current':'GDN'}
def fmt(v):
    return f'{100*st.mean(v):.2f}' + (f' ± {100*st.stdev(v):.2f}' if len(v)>1 else '')
report = ['# Database selection at 500 steps', '',
          'Mean ± sample SD across seeds; all accuracy values in percent. '
          'Held-out queries use unseen combinations of known attribute values. '
          'Balanced accuracy gives equal weight to YES and NO recall. '
          'Always-NO balanced accuracy is 50%; raw conjunction accuracy is about 93.75%.', '']
for (loss_mode,width,load,length), runs in sorted(groups.items()):
    cells = defaultdict(list)
    for r in runs: cells[(r['family'],r['d'])].append(r)
    report += [f'## {loss_mode} loss, {load} records, width{width}, T{length}', '']
    for prefix, split in [('', 'Seen combinations'), ('heldout_', 'Held-out combinations')]:
        report += [f'### {split}', '',
          '| Model | Seeds | Single BA | Conjunction BA | Conjunction exact set | Nonempty conjunction exact | Conjunction F1 | Paired conjunction BA gain (pp) |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
        lookup={(r['family'],r['d'],r['seed']):r for r in runs}
        for (family,d), rr in sorted(cells.items()):
            label=labels[family]+(f' + SMat d{d}' if d>1 else '')
            keys=['single_balanced_accuracy','conjunction_balanced_accuracy','conjunction_exact_set_accuracy',
                  'conjunction_nonempty_exact_set_accuracy','conjunction_micro_f1']
            scores=[fmt([r[prefix+k] for r in rr]) for k in keys]
            bases=[lookup.get((family,1,r['seed'])) for r in rr]
            gain=fmt([r[prefix+keys[1]]-b[prefix+keys[1]] for r,b in zip(rr,bases)]) if d>1 and all(bases) else '—'
            report.append(f'| {label} | {len(rr)} | '+ ' | '.join(scores)+f' | {gain} |')
        controls={r['seed']:r[prefix+'conjunction_always_no_exact_set_accuracy'] for r in runs}
        report += ['',f'Always-NO conjunction exact-set baseline: {fmt(list(controls.values()))}%.', '']
    report += ['| Model | Total / trainable parameters |', '|---|---:|']
    for (family,d), rr in sorted(cells.items()):
        label=labels[family]+(f' + SMat d{d}' if d>1 else '')
        report.append(f'| {label} | {rr[0]["parameters"]:,} / {rr[0]["trainable_parameters"]:,} |')
    fig, axes = plt.subplots(1,2,figsize=(10,4.5))
    for ax, family in zip(axes,['mamba2','gdn_current']):
        ds=sorted(d for f,d in cells if f==family)
        for offset,prefix,label,color in [(-.15,'','Seen','#0072B2'),(.15,'heldout_','Held out','#D55E00')]:
            vv=[[100*r[prefix+'conjunction_balanced_accuracy'] for r in cells[(family,d)]] for d in ds]
            ax.bar([d+offset for d in ds],[st.mean(v) for v in vv],width=.3,color=color,label=label,
                   yerr=[st.stdev(v) if len(v)>1 else 0 for v in vv],capsize=3)
        ax.axhline(50,color='gray',ls='--',label='Always NO')
        ax.set(title=labels[family],ylabel='Conjunction balanced accuracy (%)',ylim=(0,101),xticks=ds,
               xticklabels=['Baseline' if d==1 else f'SMat d{d}' for d in ds])
        ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2)
    counts=[len(v) for v in cells.values()]
    n=str(min(counts)) if min(counts)==max(counts) else f'{min(counts)}–{max(counts)}'
    fig.suptitle(f'Database selection: {loss_mode} loss, {load} records, T{length}, width{width}, 500 steps\nn={n} seeds; error bars show sample SD')
    fig.tight_layout()
    name=f'{loss_mode}-w{width}-r{load}-t{length}'
    fig.savefig(root/f'{name}.png',dpi=170);fig.savefig(root/f'{name}.pdf');plt.close(fig)
    report += ['',f'![{name}]({name}.png)','']
(root/'report.md').write_text('\n'.join(report)+'\n')
print(f'Collected {len(rows)} completed runs')

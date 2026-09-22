"""Final-epoch optimized-backend results, kept separate from dense runs."""
import csv
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent.parent/'results/paper_fast_20260920'
tasks=json.loads((root/'tasks.json').read_text())
rows=[]
lines=['# Fast joint-recall Log-Linear results','',
       'Final epoch 32. Accuracy in percent. Validation is the equal-cell metric used in Table 3; final-test accuracy is separate. These are fresh optimized-backend runs, not resumed dense runs.','',
       '| Backbone | Seed | Final validation | Final test | Status |',
       '|---|---:|---:|---:|---|']
for task in tasks:
    path=Path(task['result'])
    if path.exists():
        result=json.loads(path.read_text())
        assert result['complete'] and result['epochs']==32 and result['steps']==22560,path
        history=[json.loads(line) for line in (path.parent/'metrics.jsonl').read_text().splitlines()]
        final,=[r for r in history if r['epoch']==32]
        row=dict(family=task['family'],seed=task['seed'],backend=task['backend'],
                 validation_accuracy=100*final['accuracy'],test_accuracy=100*result['final_test']['accuracy'],
                 epochs=result['epochs'],steps=result['steps'],source=str(path))
        rows.append(row)
        lines.append(f"| {task['family']} | {task['seed']} | {row['validation_accuracy']:.2f} | {row['test_accuracy']:.2f} | Complete |")
    else:
        status=path.parent/'status.json'
        detail='Pending'
        if status.exists():
            s=json.loads(status.read_text());detail=f"Training: {s.get('steps',0)}/22560 steps"
        lines.append(f"| {task['family']} | {task['seed']} | — | — | {detail} |")
summary=[]
for family in ['mamba2','gdn']:
    group=[r for r in rows if r['family']==family]
    if group:
        values=[r['validation_accuracy'] for r in group]
        summary.append(dict(family=family,n=len(group),seeds=[r['seed'] for r in group],mean=statistics.mean(values),std=statistics.stdev(values) if len(group)>1 else None))
        if len(group)==5:lines+=['',f"{family}: five-seed validation mean {statistics.mean(values):.2f} ± {statistics.stdev(values):.2f} (sample SD)."]
lines.insert(2,f'{len(rows)}/10 runs complete.')
diagnostic=root/'diagnostics/fast-checkpoint-full-validation-gdn.json'
if diagnostic.exists():
    check=json.loads(diagnostic.read_text())
    lines+=['',f"GDN seed 123 diagnostic: the same final fast checkpoint scores {100*check['dense']['accuracy']:.3f}% with dense evaluation and {100*check['tree64']['accuracy']:.3f}% with fast evaluation. The low score persists under both evaluators. The cause of the training divergence from the independent dense run is unresolved; the seed remains in the aggregate."]
(root/'report.md').write_text('\n'.join(lines)+'\n')
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'per_seed.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=['family','seed','backend','validation_accuracy','test_accuracy','epochs','steps','source'])
    writer.writeheader();writer.writerows(rows)
print(f'{len(rows)}/10 fast runs complete')

# One wide table for the user's per-seed view. Joint Log-Linear comes entirely
# from the fast campaign; dense joint repeats remain separately available.
groups={}
with (root.with_name('paper_seeds_20260920')/'per_seed.csv').open() as stream:
    for row in csv.DictReader(stream):
        if row['split']!='validation' or row['cell']!='macro' or row['task'] not in ['mqar','joint']:continue
        key=(row['task'],row['family'],int(row['width']),int(row['d']))
        groups.setdefault(key,{})[int(row['seed'])]=float(row['accuracy'])
for row in rows:
    family='gdn_current' if row['family']=='gdn' else row['family']
    groups.setdefault(('joint',family,64,0),{})[row['seed']]=row['validation_accuracy']
wide=[]
for task in ['mqar','joint']:
    for family in (['gdn','mamba2'] if task=='mqar' else ['gdn_current','mamba2']):
        for width in ([16,32,64] if task=='mqar' else [64]):
            for d in [1,0,2,3,4]:
                scores=groups.get((task,family,width,d),{})
                variant={1:'Native',0:'Log-Linear (fast)' if task=='joint' else 'Log-Linear'}.get(d,f'SMat d={d}')
                row=dict(task=task,backbone='GDN' if family.startswith('gdn') else 'Mamba-2',width=width,variant=variant)
                row.update({f'seed_{seed}':scores.get(seed) for seed in [123,1,2,3,4]})
                row.update(mean=statistics.mean(scores.values()) if len(scores)==5 else None,
                           sample_sd=statistics.stdev(scores.values()) if len(scores)==5 else None)
                wide.append(row)
with (root/'all_seeds.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(wide[0]));writer.writeheader();writer.writerows(wide)
table=['# All seeds: Tables 2 and 3 with fast joint Log-Linear','',
       'Final-epoch validation accuracy (%). Joint recall uses equal-cell averaging. The joint Log-Linear rows use only fresh optimized-backend runs; their dense counterparts are reported separately. Mean and sample SD appear only after all five seeds finish.','',
       '| Task | Backbone | Width | Variant | Seed 123 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± SD |',
       '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
for row in wide:
    fields=[row['task'],row['backbone'],str(row['width']),row['variant']]
    fields += ['Pending' if row[f'seed_{seed}'] is None else f"{row[f'seed_{seed}']:.2f}" for seed in [123,1,2,3,4]]
    fields += ['Pending' if row['mean'] is None else f"{row['mean']:.2f} ± {row['sample_sd']:.2f}"]
    table.append('| '+' | '.join(fields)+' |')
(root/'all_seeds.md').write_text('\n'.join(table)+'\n')
tex=[r'\begin{tabular}{lcc}',r'\toprule',r'Variant & Mamba-2 & GDN \\',r'\midrule']
for d,label in [(1,'Native'),(0,'Log-Linear (optimized)'),(2,r'+ SMat ($d=2$)'),(3,r'+ SMat ($d=3$)'),(4,r'+ SMat ($d=4$)')]:
    cells=[]
    for family in ['mamba2','gdn_current']:
        scores=groups.get(('joint',family,64,d),{})
        cells.append(f'${statistics.mean(scores.values()):.2f} \\pm {statistics.stdev(scores.values()):.2f}$' if len(scores)==5 else r'$\text{pending}$')
    tex.append(label+' & '+' & '.join(cells)+r' \\')
tex += [r'\bottomrule',r'\end{tabular}']
(root/'table3_fast.tex').write_text('\n'.join(tex)+'\n')

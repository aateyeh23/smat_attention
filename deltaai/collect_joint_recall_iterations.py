"""Report every scheduled development run, emphasizing validation for selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'results/joint_recall_iterations'
tasks=[t for t in json.loads((ROOT/'campaign.json').read_text()) if t.get('kind','training')=='training'];trials={};rows=[]
for t in tasks:
    folder=Path(t['result']).parent;trial=folder.parent.name
    if not (folder/'recipe.json').exists():continue
    recipe=json.loads((folder/'recipe.json').read_text())
    h=[]
    if (folder/'metrics.jsonl').exists():
        for line in (folder/'metrics.jsonl').read_text().splitlines():
            try:h.append(json.loads(line))
            except json.JSONDecodeError:pass
    last=h[-1] if h else {};best=max(h,key=lambda r:r['accuracy']) if h else {}
    status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    result=json.loads(Path(t['result']).read_text()) if Path(t['result']).exists() else {}
    row=dict(trial=trial,condition=recipe['condition'],family=recipe['family'],d=recipe['d'],
             dataset=(folder.parent/'data').resolve().parent.name,
             dataset_manifest_sha256=recipe['dataset_manifest_sha256'],
             seed=recipe['seed'],lr=recipe['lr'],complete=bool(result),epoch=last.get('epoch',0),
             weight_decay=recipe['weight_decay'],memory_timescale=recipe.get('memory_timescale_initialization',0),
             gdn_memory_scalar_decay=recipe.get('gdn_memory_scalar_decay',True) if recipe['family']=='gdn_current' and recipe['d']>=2 else None,
             steps=status.get('steps',0),validation=last.get('accuracy'),best_validation=best.get('accuracy'),
             validation_cells=last.get('cells',{}),
             best_epoch=best.get('epoch'),final_test=result.get('final_test',{}).get('accuracy'))
    rows.append(row);trials.setdefault(trial,[]).append((recipe,h))
report=['# Joint-recall development','',
        'Single seed123. All trials are retained. Validation guides iteration; '
        'the selected configuration still needs a fresh confirmation test. '
        'Incomplete training results are not final comparisons.','']
groups={}
for row in rows:
    if row['complete'] and row['condition']=='shared':
        groups.setdefault((row['dataset_manifest_sha256'],row['family']),[]).append(row)
report+=['## Strongest completed settings on identical data','',
         'Each entry selects the highest final-epoch validation score among completed '
         'settings for that backbone and dataset. Pending settings can change these '
         'comparisons. This development table is not independent confirmation.','',
         '| Dataset | Backbone | Native validation (%) | SMat validation (%) | SMat minus native (pp) | Selected trials: native / SMat |',
         '|---|---|---:|---:|---:|---|']
selected_pairs=[]
for (_,family),group in groups.items():
    native=[r for r in group if r['d']==1];smat=[r for r in group if r['d']>=2]
    if not native or not smat:continue
    native=max(native,key=lambda r:r['validation']);smat=max(smat,key=lambda r:r['validation'])
    selected_pairs.append((native,smat))
    report.append(f'| {native["dataset"]} | {family} | {native["validation"]*100:.2f} | '
                  f'{smat["validation"]*100:.2f} | {(smat["validation"]-native["validation"])*100:+.2f} | '
                  f'{native["trial"]} / {smat["trial"]} (d={smat["d"]}) |')
datasets=list(dict.fromkeys(n['dataset'] for n,_ in selected_pairs))
if datasets:
    load_fig,load_axes=plt.subplots(len(datasets),2,figsize=(12,3.5*len(datasets)),squeeze=False)
    plotted=set();selected_cells=[]
    for native,smat in selected_pairs:
        row=datasets.index(native['dataset']);col=0 if native['family']=='mamba2' else 1
        ax=load_axes[row,col];plotted.add((row,col));cells=list(native['validation_cells'])
        assert set(cells)==set(smat['validation_cells'])
        shapes=[tuple(int(part[1:]) for part in cell.split('-')) for cell in cells]
        loads=[c*k for c,k in shapes]
        smat_label=f'SMat d={smat["d"]}'+(' (no scalar decay)' if smat['gdn_memory_scalar_decay'] is False else '')
        for record,label,color in [(native,'Native','#333333'),(smat,smat_label,'#D55E00')]:
            ax.plot(loads,[record['validation_cells'][cell]['accuracy']*100 for cell in cells],
                    'o-',color=color,label=label)
        ax.set_xscale('log',base=2)
        ax.set_xticks(loads,[f'{load}\n({c}×{k})' for load,(c,k) in zip(loads,shapes)])
        family='Mamba-2' if col==0 else 'Gated DeltaNet'
        ax.set(title=f'{native["dataset"]}\n{family}',ylabel='Final validation accuracy (%)',
               xlabel='Bindings (contexts × keys)',ylim=(0,103))
        ax.axhline(6.25,color='#aaaaaa',ls=':',lw=1);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
        for cell in cells:
            selected_cells.append(dict(dataset=native['dataset'],family=native['family'],cell=cell,
                native_trial=native['trial'],smat_trial=smat['trial'],smat_d=smat['d'],
                native_accuracy=native['validation_cells'][cell]['accuracy'],
                smat_accuracy=smat['validation_cells'][cell]['accuracy']))
    for row in range(len(datasets)):
        for col in range(2):
            if (row,col) not in plotted:load_axes[row,col].set_visible(False)
    load_fig.suptitle('Completed validation comparisons · seed 123 · selected settings')
    load_fig.tight_layout();load_fig.savefig(ROOT/'final_validation_by_load.png',dpi=160)
    load_fig.savefig(ROOT/'final_validation_by_load.pdf');plt.close(load_fig)
    (ROOT/'selected_validation_cells.json').write_text(json.dumps(selected_cells,indent=2)+'\n')
    report+=['','Every load is shown below for the selected completed settings. '
             'Datasets occupy separate panels; their training regimes are not interchangeable.',
             '','![Completed validation by table size](final_validation_by_load.png)','']
primary=[r for r in rows if r['trial']=='explicit_context_capacity512' and r['condition']=='shared']
if primary:
    report+=['','## Capacity sweep at the common learning rate','',
             'All dimensions below use LR0.003, weight decay0.1, width64, two layers, and seed123. '
             'Only completed 32-epoch runs appear in this table and plot; partial curves remain below.',
             '','| Backbone | Variant | 4 bindings | 16 bindings | 128 bindings | 256 bindings | 512 bindings | Macro (%) |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    capacity_fig,capacity_axes=plt.subplots(1,2,figsize=(12,4))
    palette={1:'#333333',2:'#0072B2',3:'#D55E00',4:'#009E73'}
    for ax,family in zip(capacity_axes,['mamba2','gdn_current']):
        for row in sorted(primary,key=lambda r:r['d']):
            if row['family']!=family or not row['complete']:continue
            cells=list(row['validation_cells']);loads=[int(c.split('-')[0][1:])*int(c.split('-')[1][1:]) for c in cells]
            scores=[100*row['validation_cells'][cell]['accuracy'] for cell in cells]
            label='Native' if row['d']==1 else f'SMat d={row["d"]}'
            ax.plot(loads,scores,'o-',label=label,color=palette[row['d']])
            report.append(f'| {family} | {label} | '+' | '.join(f'{v:.2f}' for v in scores)+f' | {100*row["validation"]:.2f} |')
        oracle_path=ROOT/'explicit_context_capacity512/validation_oracle_summary.json'
        if oracle_path.exists():
            oracle=json.loads(oracle_path.read_text())['cells']
            loads=[int(c['cell'].split('-')[0][1:])*int(c['cell'].split('-')[1][1:]) for c in oracle]
            ax.plot(loads,[100*c['best_single_field'] for c in oracle],'--',color='#999999',label='Best single-field oracle')
        ax.set_xscale('log',base=2);ax.set_xticks([4,16,128,256,512],['4','16','128','256','512'])
        ax.set(title=family,xlabel='Bindings',ylabel='Final validation accuracy (%)',ylim=(0,103))
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    capacity_fig.tight_layout();capacity_fig.savefig(ROOT/'capacity_all_d_validation.png',dpi=180)
    capacity_fig.savefig(ROOT/'capacity_all_d_validation.pdf');plt.close(capacity_fig)
    report+=['','![Common-recipe capacity sweep](capacity_all_d_validation.png)','']
report+=['','## All development runs','',
        '| Trial | Condition | Model | Epoch | Final/latest validation (%) | Best validation (%) | Complete |',
        '|---|---|---|---:|---:|---:|---|']
def pct(value):return '—' if value is None else f'{100*value:.2f}'
for r in rows:
    label=r['family']+(' native' if r['d']==1 else f' SMat d{r["d"]}')
    report.append(f'| {r["trial"]} | {r["condition"]} | {label} | {r["epoch"]} | {pct(r["validation"])} | '
                  f'{pct(r["best_validation"])} | {r["complete"]} |')
fig,axes=plt.subplots(len(trials),1,figsize=(10,4*max(1,len(trials))),squeeze=False)
colors={('mamba2',1):'#333333',('mamba2',2):'#CC79A7',('mamba2',3):'#D55E00',('mamba2',4):'#E69F00',
        ('gdn_current',1):'#0072B2',('gdn_current',2):'#56B4E9',('gdn_current',3):'#009E73',('gdn_current',4):'#66AA55'}
for ax,(trial,records) in zip(axes[:,0],trials.items()):
    for q,h in records:
        if not h:continue
        label=q['family']+(' native' if q['d']==1 else f' SMat d={q["d"]}')+' '+q['condition']
        ax.plot([v['epoch'] for v in h],[v['accuracy']*100 for v in h],label=label,
                color=colors[q['family'],q['d']],ls='-' if q['condition']=='shared' else '--')
    ax.set(title=trial,xlabel='Epoch',ylabel='Validation accuracy (%)',xlim=(0,32),ylim=(0,101))
    ax.grid(alpha=.2)
    if any(h for _,h in records):ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'learning_curves.png',dpi=150);fig.savefig(ROOT/'learning_curves.pdf');plt.close(fig)
report+=['','![Validation learning curves](learning_curves.png)','']
(ROOT/'report.md').write_text('\n'.join(report)+'\n')
(ROOT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(f'{sum(r["complete"] for r in rows)}/{len(tasks)} scheduled runs complete')

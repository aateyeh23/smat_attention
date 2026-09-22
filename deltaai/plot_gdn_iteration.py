"""Export an audited iteration's learning curve, matched slices, and tables."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def export(root):
    expected=json.loads((root/'expected-sources.json').read_text())
    actual=json.loads((root/'training-sources.json').read_text())
    if actual!=expected:
        raise ValueError('Training sources are not verified')
    baseline={r['epoch']:r['metrics'] for r in map(json.loads,(root/'baseline.jsonl').read_text().splitlines())}
    history={r['epoch']:r['metrics'] for r in map(json.loads,(root/'w32-d3-history.jsonl').read_text().splitlines())}
    gates=[json.loads(s) for s in (root/'gates.jsonl').read_text().splitlines()]
    latest=max(set(history)&{r['epoch'] for r in gates})
    gates=[r for r in gates if r['epoch']<=latest]
    required=[r for r in gates if r['required']]
    passed=[r['epoch'] for r in required if r['decision']=='pass']
    all_required=list(range(4,33,2))
    result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {}
    summary=dict(trial=root.name,source_verified=True,epoch=latest,
                 accuracy=history[latest]['valid/accuracy'],baseline=baseline[latest]['valid/accuracy'],
                 margin=history[latest]['valid/accuracy']-baseline[latest]['valid/accuracy'],
                 passed_gates=passed,remaining_gates=[e for e in all_required if e not in passed],
                 goal_achieved=(passed==all_required and result.get('epoch')==32
                                and result.get('decision')=='pass' and result.get('complete') is True))
    (root/'run-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'gate-comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','accuracy','baseline','margin','decision'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in required)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'}):
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),gridspec_kw={'width_ratios':[1.2,1]})
        b_epochs=sorted(baseline)
        epochs=[r['epoch'] for r in gates]
        axes[0].plot(b_epochs,[100*baseline[e]['valid/accuracy'] for e in b_epochs],
                     color='#777d88',lw=2,label='GDN')
        axes[0].plot(epochs,[100*r['accuracy'] for r in gates],color='#147d76',lw=2,label='GDN + SMAT')
        axes[0].scatter([r['epoch'] for r in required],[100*r['accuracy'] for r in required],
                        s=24,color=['#147d76' if r['decision']=='pass' else '#bd3d36' for r in required],zorder=3)
        axes[0].set(xlabel='Epoch',ylabel='Validation accuracy (%)',xlim=(1,32),ylim=(0,100),
                    title=f'Learning curve · {len(passed)}/15 gates passed')
        axes[0].legend(frameon=False,loc='lower right');axes[0].grid(axis='y',alpha=.2)
        pairs=[4,8,16,32,64];x=np.arange(len(pairs));width=.36
        key=lambda n:f'valid/num_kv_pairs/accuracy-{n}'
        axes[1].bar(x-width/2,[100*baseline[latest][key(n)] for n in pairs],width,color='#a0a5ae',label='GDN')
        axes[1].bar(x+width/2,[100*history[latest][key(n)] for n in pairs],width,color='#147d76',label='GDN + SMAT')
        axes[1].set(xticks=x,xticklabels=pairs,xlabel='Key–value pairs',ylim=(0,100),
                    title=f'Matched evaluation cells · epoch {latest}')
        axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
        fig.suptitle('MQAR: width 32 · two heads · head/state dimension 16 · SMAT d=3',fontsize=12,y=.98)
        fig.text(.5,.025,'Seed 123 · one write and four weighted reads per head · runtime sources verified',
                 ha='center',fontsize=9,color='#505761')
        fig.tight_layout(rect=(0,.065,1,.93))
        fig.savefig(root/'comparison.png',dpi=180,bbox_inches='tight')
        fig.savefig(root/'comparison.pdf',bbox_inches='tight');plt.close(fig)
    selected=[4,8,16,24,32]
    fmt=lambda rows,e:(f'{100*rows[e]["valid/accuracy"]:.2f}' if e in rows else '--')
    lines=[r'% Requires \usepackage{booktabs}',r'\begin{tabular}{lrrrrr}',r'\toprule',
           'Model & '+ ' & '.join(f'Epoch {e}' for e in selected)+r' \\',r'\midrule',
           'GDN & '+' & '.join(fmt(baseline,e) for e in selected)+r' \\',
           r'GDN + SMAT ($d=3$) & '+' & '.join(fmt(history,e) for e in selected)+r' \\',
           r'\bottomrule',r'\end{tabular}',
           '% Accuracy in percent; width32, two heads, head/state16, seed123.']
    (root/'comparison.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trial',default='22_verified_transport')
    args=parser.parse_args()
    export(Path(__file__).resolve().parent/'results/mqar_gdn_iterations'/args.trial)

"""Export the matched-epoch accuracy curves for all five width/family groups."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent/'results/mqar_width_sweeps'
fig, axes = plt.subplots(2, 5, figsize=(18, 6.8), sharex=True, sharey='row')
groups = [('gdn',32),('gdn',64),('mamba2',16),('mamba2',32),('mamba2',64)]
colors = ['#333333','#0072B2','#D55E00','#009E73']
labels = ['Baseline','SMAT d=2','SMAT d=3','SMAT d=4']
for col, (family, width) in enumerate(groups):
    for d, color, label in zip((1,2,3,4), colors, labels):
        path = root/family/f'w{width}-d{d}-history.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(s) for s in path.read_text().splitlines() if s]
        for row, metric in enumerate(('valid/accuracy','valid/num_kv_pairs/accuracy-64')):
            axes[row,col].plot([x['epoch'] for x in rows], [100*x['metrics'][metric] for x in rows],
                               color=color,label=label,linewidth=1.7)
    axes[0,col].set_title(f'{family.upper()} · width {width}')
    axes[1,col].set_xlabel('Epoch')
    for ax in axes[:,col]:
        ax.set_xlim(1,32)
        ax.set_ylim(0,100)
        ax.grid(alpha=.2)
        ax.spines[['right','top']].set_visible(False)
axes[0,0].set_ylabel('Overall accuracy (%)')
axes[1,0].set_ylabel('64-pair accuracy (%)')
handles, names = axes[0,0].get_legend_handles_labels()
fig.legend(handles,names,loc='lower center',ncol=4,frameon=False)
fig.suptitle('MQAR · one learned write, four distinct weighted reads · seed 123', y=.99)
fig.tight_layout(rect=(0,.055,1,.955))
fig.savefig(root/'accuracy_curves.png',dpi=180)
fig.savefig(root/'accuracy_curves.pdf')
print(root/'accuracy_curves.png')

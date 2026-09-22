"""Static matched-epoch plots for the single-arm transport comparison."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'results'
OUT=ROOT/'mqar_gdn_transport/w32_d3_s123'
parser=argparse.ArgumentParser()
parser.add_argument('--neighbor-write-grad',action='store_true')
args=parser.parse_args()
series=[('GDN',ROOT/'mqar_width_sweeps/gdn/w32-d1-history.jsonl'),
        ('Original SMAT d=3',ROOT/'mqar_width_sweeps/gdn/w32-d3-history.jsonl'),
        ('Transport SMAT d=3',OUT/'w32-d3-history.jsonl')]
if args.neighbor_write_grad:
    OUT=OUT.with_name(OUT.name+'_neighbor_grad')
    series.append(('Transport + neighbor gradients',OUT/'w32-d3-history.jsonl'))
fig,axes=plt.subplots(1,3,figsize=(12,3.6))
metrics=[('valid/accuracy','Overall accuracy (%)',100),
         ('valid/num_kv_pairs/accuracy-64','64-pair accuracy (%)',100),
         ('valid/loss','Validation loss',1)]
for label,path in series:
    history=[json.loads(s) for s in path.read_text().splitlines()]
    for ax,(key,title,scale) in zip(axes,metrics):
        ax.plot([r['epoch'] for r in history],[scale*r['metrics'][key] for r in history],label=label,lw=2)
        ax.set(xlabel='Epoch',ylabel=title,xlim=(1,32));ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('MQAR: width 32, 2 heads, head/state 16; seed 123, LR 0.01')
fig.tight_layout()
for ext in ('png','pdf'): fig.savefig(OUT/f'comparison.{ext}',dpi=180)

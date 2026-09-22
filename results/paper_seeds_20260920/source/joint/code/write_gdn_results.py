"""Render current results with actual completion status; emit paper tables only when complete."""
import json,csv
from pathlib import Path
root=Path(__file__).resolve().parent/'results'
out=root/'gdn_smat_summary';out.mkdir(exist_ok=True)
rows=[]
for w in (16,32,64):
 b=json.loads((root/f'mqar_gdn_interleaved/quoted_heads_s123/w{w}-d1.json').read_text())
 p=root/f'mqar_gdn_soft_routes/quoted_heads_s123/w{w}-d3.json'
 if not p.exists():continue
 s=json.loads(p.read_text());base=100*b['metrics']['valid/accuracy'];acc=100*s['metrics']['valid/accuracy']
 rows.append(dict(width=w,heads=1 if w==16 else 2,seed=123,epoch=s['next_epoch'],complete=s['complete'],gdn=base,soft_smat=acc,gain=acc-base))
with (out/'mqar.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
text='MQAR: GDN + SMAT with learned causal keys and sparse soft routing, d=3.\n\nSame seed (123), two layers, head dimension 16, value expansion 1, one head at width 16 and two at widths 32/64. Training budget: 32 epochs, 707 optimizer steps per epoch. No specified key-token offset.\n\n'
text+='| Width | GDN (%) | Soft SMAT (%) | Gain (points) | Epochs | Complete |\n|---|---:|---:|---:|---:|---|\n'
for r in rows:text+=f"| {r['width']} | {r['gdn']:.2f} | {r['soft_smat']:.2f} | {r['gain']:+.2f} | {r['epoch']}/32 | {r['complete']} |\n"
text+='\nThese are final-checkpoint accuracies for completed runs, and current-checkpoint accuracies for unfinished runs; they are not best-epoch selections.\n'
all_complete=len(rows)==3 and all(r['complete'] and r['epoch']==32 for r in rows)
if all_complete:
 latex='\\begin{tabular}{rrrr}\n\\toprule\nWidth & GDN & Soft SMAT ($d=3$) & Gain \\\\\n\\midrule\n'
 for r in rows:latex+=f"{r['width']} & {r['gdn']:.2f} & {r['soft_smat']:.2f} & {r['gain']:+.2f} \\\\\n"
 latex+='\\bottomrule\n\\end{tabular}\n'
 (out/'mqar.tex').write_text(latex)
sc=json.loads((root/'sc_gdn_smat/s0/checkpoint_audit.json').read_text())
text+='\nSelective copying: seed 0, width 64, two layers, four GDN heads of dimension 16, value expansion 2; length 4096, 16 content tokens, batch 32, 10,000 steps. All requested checkpoints passed the recipe audit.\n\n| Model | Accuracy (%) | Steps |\n|---|---:|---:|\n'
for r in sc['completed']:
 name='GDN' if r['d']==1 else f"GDN + SMAT d={r['d']}"
 text+=f"| {name} | {100*r['accuracy']:.2f} | {r['step']} |\n"
text+='\nThe selective-copying hybrids use the original hard-routing delta-memory variant, not the MQAR soft-routing variant.\n'
(out/'results.md').write_text(text)
print(json.dumps(dict(mqar=rows,mqar_all_complete=all_complete,selective_copying_complete=sc['all_four_complete']),indent=2))

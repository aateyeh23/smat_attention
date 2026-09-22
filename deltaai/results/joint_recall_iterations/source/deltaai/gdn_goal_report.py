"""Compare completed matched-recipe MQAR runs; never mix legacy head counts."""
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'results'
rows=[]
for width in (16,32,64):
    entries={}
    for name,group,d in [('gdn','mqar_gdn_interleaved',1),('additive_d3','mqar_gdn_interleaved',3),('delta_d3','mqar_gdn_delta',3),('aligned_d3','mqar_gdn_aligned',3),('aligned_soft_d3','mqar_gdn_aligned_soft',3),('tied_keys_d3','mqar_gdn_tied_keys',3),('causal_keys_d3','mqar_gdn_causal_keys',3),('causal_soft_d3','mqar_gdn_causal_soft',3),('route_grad_d3','mqar_gdn_route_grad',3),('soft_routes_d3','mqar_gdn_soft_routes',3)]:
        path=ROOT/group/'quoted_heads_s123'/f'w{width}-d{d}.json'
        if path.exists():
            x=json.loads(path.read_text())
            entries[name]=dict(epoch=x['next_epoch'],complete=x['complete'],accuracy=x['metrics']['valid/accuracy'],path=str(path),
                               pruned=path.with_name(path.stem+'-pruned.json').exists())
    baseline=entries.get('gdn',{})
    wins=[name for name,e in entries.items() if name not in ('gdn','aligned_d3','aligned_soft_d3','tied_keys_d3') and e['complete'] and baseline.get('complete') and e['accuracy']>baseline['accuracy']]
    rows.append(dict(width=width,heads=1 if width==16 else 2,runs=entries,verified_winners=wins))
report=dict(mqar=rows,mqar_all_widths_beaten=all(r['verified_winners'] for r in rows))
sc=[]
for d in (1,2,3,4):
    path=ROOT/'sc_gdn_smat/s0'/f'w64-d{d}.json'
    row=dict(d=d,complete=False,step=0)
    if path.exists():
        x=json.loads(path.read_text())
        row.update(step=x['step'],complete=x['complete'],seed=x['args']['seed'],path=str(path))
        log=path.with_suffix('.log')
        if log.exists():
            matches=re.findall(r'^(?:EVAL|FINAL) step (\d+) acc ([0-9.]+)',log.read_text(),re.M)
            if matches: row.update(eval_step=int(matches[-1][0]),accuracy=float(matches[-1][1]))
    sc.append(row)
report['selective_copying']=sc
report['consistent_soft_routing_all_widths_beaten']=all('soft_routes_d3' in r['verified_winners'] for r in rows)
report['all_requested_runs_successful']=report['consistent_soft_routing_all_widths_beaten'] and all(r['complete'] for r in sc)
print(json.dumps(report,indent=2))

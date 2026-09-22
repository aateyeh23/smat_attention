"""Check completed SC checkpoints against the matched Mamba data/training recipe."""
import json,re
from pathlib import Path
import torch
root=Path(__file__).resolve().parent/'results/sc_gdn_smat/s0'
expected=dict(d_model=64,n_layers=2,seq_len=4096,n_copy=16,n_vocab=16,batch=32,steps=10000,lr=.001,warmup=500,wd=.1,seed=0,gdn_headdim=16,eval_every=250,ckpt_every=250)
rows=[]
for d in (1,2,3,4):
 p=root/f'w64-d{d}.json'
 if not p.exists():continue
 status=json.loads(p.read_text())
 if not status['complete']:continue
 ck=torch.load(p.with_suffix('.pt'),map_location='cpu',weights_only=False)
 assert ck['step']==status['step']==10000
 for k,v in expected.items():assert ck['args'][k]==v,(d,k,ck['args'][k],v)
 assert ck['args']['arm']==('gdn' if d==1 else 'gdn_smat')
 assert ck['args']['d']==d
 for suffix,shape in [('gdn.q_proj.weight',(64,64)),('gdn.v_proj.weight',(128,64)),('gdn.b_proj.weight',(4,64))]:
  weights=[v for k,v in ck['model'].items() if k.endswith(suffix)]
  assert len(weights)==2 and all(tuple(v.shape)==shape for v in weights),(d,suffix)
 steps=[int(v['step']) for v in ck['opt']['state'].values() if 'step' in v]
 assert max(steps)==10000
 matches=re.findall(r'^FINAL step 10000 acc ([0-9.]+)',p.with_suffix('.log').read_text(),re.M)
 assert matches
 rows.append(dict(d=d,step=10000,accuracy=float(matches[-1]),head_dim=16,heads=4,expand_v=2,matched_recipe=True))
 print(rows[-1],flush=True)
(root/'checkpoint_audit.json').write_text(json.dumps(dict(completed=rows,all_four_complete=len(rows)==4),indent=2))

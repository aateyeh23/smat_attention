"""Audit completed MQAR checkpoints against the requested heads and training recipe."""
import json,re
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent/'results'
GROUPS=['mqar_gdn_interleaved','mqar_gdn_delta','mqar_gdn_aligned','mqar_gdn_aligned_soft','mqar_gdn_tied_keys','mqar_gdn_causal_keys','mqar_gdn_causal_soft','mqar_gdn_route_grad','mqar_gdn_soft_routes']
rows=[]
for width in (16,32,64):
 heads=1 if width==16 else 2
 for group in GROUPS:
  folder=ROOT/group/'quoted_heads_s123'
  for d in ((1,3) if group=='mqar_gdn_interleaved' else (3,)):
   status_path=folder/f'w{width}-d{d}.json'
   if not status_path.exists(): continue
   status=json.loads(status_path.read_text())
   if not status.get('complete'): continue
   ck=torch.load(status_path.with_suffix('.pt'),map_location='cpu',weights_only=False)
   assert ck['complete'] and ck['next_epoch']==status['next_epoch']
   assert ck['metrics']==status['metrics']
   q=[v for k,v in ck['model'].items() if k.endswith('gdn.q_proj.weight')]
   v=[v for k,v in ck['model'].items() if k.endswith('gdn.v_proj.weight')]
   a=[v for k,v in ck['model'].items() if k.endswith('gdn.a_proj.weight')]
   assert len(q)==len(v)==len(a)==2
   assert all(tuple(t.shape)==(heads*16,width) for t in q+v)
   assert all(tuple(t.shape)==(heads,width) for t in a)
   epochs=ck['next_epoch'];expected_steps=epochs*707
   steps=[int(s['step']) for s in ck['optimizer']['state'].values() if 'step' in s]
   assert max(steps)==expected_steps,(group,width,epochs,max(steps))
   assert ck['scheduler']['T_max']==32
   assert epochs==32 or ck['metrics']['valid/accuracy']>.99
   log=status_path.with_suffix('.log').read_text()
   seeds=set(re.findall(r'\bseed=(\d+)',log))
   assert seeds=={'123'},seeds
   if group=='mqar_gdn_soft_routes':
    assert "'memory_route_soft': True" in log
    assert "'memory_key_mode': 'tied_causal'" in log
    assert "'hash_key_shift': 1" not in log
    assert "'memory_update': 'delta'" in log
    mq=[v for k,v in ck['model'].items() if k.endswith('memory_qk.weight')]
    assert len(mq)==2 and all(tuple(t.shape)==(heads*16,width) for t in mq)
   row=dict(group=group,width=width,d=d,heads=heads,layers=2,head_dim=16,
            seed=123,epochs=epochs,optimizer_steps=max(steps),accuracy=status['metrics']['valid/accuracy'])
   rows.append(row);print(json.dumps(row),flush=True)
report=dict(checkpoints=rows,all_completed_checkpoints_passed=True)
(ROOT/'mqar_checkpoint_audit.json').write_text(json.dumps(report,indent=2))

"""Read-only first-layer feature/routing probe on the aligned MQAR checkpoint."""
import os,json
from pathlib import Path
os.environ['ZOO_DM']='32';os.environ['ZOO_DS']='3'
import torch
from zoology.model import LanguageModel
from zoology.data.utils import prepare_data
from zoo_gdn_aligned_configs import configs
from zoo_smat_gdn import SmatGDNReset

folder=Path('results/mqar_gdn_aligned/quoted_heads_s123')
ck=torch.load(folder/'w32-d3.pt',map_location='cpu',weights_only=False)
config=configs[0]
model=LanguageModel(config.model).cuda().eval();model.load_state_dict(ck['model'])
_,test=prepare_data(config.data)
mixer=next(m for m in model.modules() if isinstance(m,SmatGDNReset))
captured={}
def capture(module,args): captured.update(ca=module,q=args[1],k=args[2],n=args[4])
for mods in mixer.ca.values():
 for ca in mods.values(): ca.register_forward_pre_hook(capture)
rows=[];seen=set()
with torch.no_grad():
 for x,y,slices in test:
  pairs=int(slices[0]['num_kv_pairs'])
  if pairs not in (16,64) or pairs in seen: continue
  seen.add(pairs);x=x.cuda();y=y.cuda();model(x)
  b,t=x.shape;h=mixer.h;n=captured['n'];ca=captured['ca']
  query_pos=(y!=-100).nonzero()[:,1].reshape(b,pairs)
  query_tokens=x.gather(1,query_pos)
  prefix_keys=x[:,:2*pairs:2]
  match=(prefix_keys[:,:,None]==query_tokens[:,None,:]).long().argmax(1)
  value_pos=match*2+1
  valid=(value_pos<n)&(query_pos>=n)
  q=captured['q'].reshape(b,h,t,-1)
  k=captured['k'].reshape(b,h,t,-1)
  q=q.gather(2,query_pos[:,None,:,None].expand(b,h,pairs,mixer.r))
  values=k[:,:,1:2*pairs:2]
  scores=torch.einsum('bhqr,bhkr->bhqk',q,values)
  matched=scores.gather(-1,match[:,None,:,None].expand(b,h,pairs,1)).squeeze(-1)
  top1=scores.argmax(-1)==match[:,None,:]
  keys=ca.last_hard_k.reshape(b,h,n).gather(2,value_pos.clamp(max=n-1)[:,None,:].expand(b,h,pairs))
  queries=ca._sparse[0][...,0].reshape(b,h,t-n).gather(2,(query_pos-n).clamp(min=0)[:,None,:].expand(b,h,pairs))
  mask=valid[:,None,:].expand(b,h,pairs)
  row=dict(pairs=pairs,query_heads=int(mask.sum()),key_query_cosine=float(matched[mask].mean()),
           correct_key_top1=float(top1[mask].float().mean()),
           correct_key_absolute_top1=float((scores.abs().argmax(-1)==match[:,None,:])[mask].float().mean()),
           correct_key_bottom1=float((scores.argmin(-1)==match[:,None,:])[mask].float().mean()),
           matched_absolute_cosine=float(matched[mask].abs().mean()),
           all_keys_absolute_cosine=float(scores.abs().mean(-1)[mask].mean()),
           same_profile=float((keys==queries)[mask].float().mean()))
  rows.append(row);print(json.dumps(row),flush=True)
  if len(seen)==2: break
result=dict(checkpoint_epoch=ck['next_epoch'],first_layer_only=True,batches_per_cell=1,rows=rows)
(folder/'key_feature_probe.json').write_text(json.dumps(result,indent=2))

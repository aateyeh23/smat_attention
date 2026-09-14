"""Re-evaluate each completed soft-routing checkpoint without optimizer updates."""
import os,json
os.environ['ZOO_DM']='32';os.environ['ZOO_DS']='3'
from pathlib import Path
from collections import defaultdict
import torch
from zoology.model import LanguageModel
from zoology.data.utils import prepare_data
from zoo_gdn_soft_routes_configs import configs
from smat.mixers.gdn import SmatGDNReset
folder=Path('results/mqar_gdn_soft_routes/quoted_heads_s123')
config=configs[0];_,test=prepare_data(config.data)
rows=[]
for width in (16,32,64):
 p=folder/f'w{width}-d3.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
 assert ck['complete'] and ck['next_epoch']==32
 config.model.d_model=width
 config.model.sequence_mixer.kwargs['n_heads']=1 if width==16 else 2
 model=LanguageModel(config.model).cuda().eval();model.load_state_dict(ck['model'])
 mixers=[m for m in model.modules() if isinstance(m,SmatGDNReset)]
 assert len(mixers)==2
 assert all(m.h==(1 if width==16 else 2) and m.memory_key_mode=='tied_causal' and m.hash_key_shift is None and m.memory_route_soft for m in mixers)
 sums=defaultdict(float);counts=defaultdict(int)
 with torch.no_grad():
  for x,y,slices in test:
   x=x.cuda();y=y.cuda();pred=model(x).argmax(-1);mask=y!=-100
   acc=((pred==y)&mask).sum(-1)/mask.sum(-1)
   for a,sl in zip(acc.tolist(),slices):
    cell=str(sl['num_kv_pairs']);sums[cell]+=a;counts[cell]+=1
 accuracy=sum(sums.values())/sum(counts.values())
 expected=ck['metrics']['valid/accuracy']
 assert abs(accuracy-expected)<1e-8,(width,accuracy,expected)
 row=dict(width=width,accuracy=accuracy,saved_accuracy=expected,epochs=32,examples=sum(counts.values()),per_cell={k:sums[k]/counts[k] for k in sums},parameter_count=sum(p.numel() for p in model.parameters()),matched=True)
 rows.append(row);print(json.dumps(row),flush=True)
 (folder/'reevaluation.json').write_text(json.dumps(dict(rows=rows,all_three_verified=len(rows)==3),indent=2))
 del model,ck;torch.cuda.empty_cache()

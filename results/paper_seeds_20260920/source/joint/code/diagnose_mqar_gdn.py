"""Read-only checkpoint interventions for the early MQAR GDN/SMAT drop."""
import os
os.environ['ZOO_DM']='16'
os.environ['ZOO_DS']='3'
import json
from pathlib import Path
from collections import defaultdict
import torch
from zoology.model import LanguageModel
from zoology.data.utils import prepare_data
from zoo_gdn_reset_configs import configs
from zoo_smat_gdn import SmatGDNReset

torch.set_num_threads(2)
folder=Path('results/mqar_gdn_reset/w16w32_s123')
outdir=folder/'diagnosis'; outdir.mkdir(exist_ok=True)
config=configs[0]
train, test=prepare_data(config.data)
print('TRAIN ORDER',[(i,len(s),s.slices) for i,s in enumerate(train.dataset.segments)],flush=True)
@torch.no_grad()
def evaluate(model, batches):
    totals=defaultdict(float); counts=defaultdict(int)
    for x,y,slices in batches:
        x=x.cuda(); y=y.cuda(); pred=model(x).argmax(-1)
        mask=y != -100
        acc=((pred==y)&mask).sum(-1)/mask.sum(-1)
        for a,sl in zip(acc.tolist(),slices):
            key=str(sl['num_kv_pairs']); totals[key]+=a; counts[key]+=1
    return dict(accuracy=sum(totals.values())/sum(counts.values()),
                per_kv={k:totals[k]/counts[k] for k in totals})
results={}
for width in (16,32):
    ck=torch.load(folder/f'w{width}-d3.pt',map_location='cpu',weights_only=False)
    config.model.d_model=width
    # This diagnostic explicitly loads the preserved, older head-count campaign.
    config.model.sequence_mixer.kwargs['n_heads']=2 * width // 16
    model=LanguageModel(config.model).cuda().eval()
    model.load_state_dict(ck['model'])
    mixers=[m for m in model.modules() if isinstance(m,SmatGDNReset)]
    row=dict(epoch=ck['next_epoch'],saved_metrics=ck['metrics'])
    for name,anneal,disable in [('baseline',1.,False),('soft_hash',0.,False),('G_off',1.,True)]:
        for m in mixers:
            m.disable_g=disable
            for mods in m.ca.values():
                for ca in mods.values(): ca.anneal=anneal
        row[name]=evaluate(model,test)
        print(width,name,row[name],flush=True)
    for m in mixers: m.disable_g=False
    # Compare a fixed subset of each training cell with the held-out cells.
    batches=[train.dataset[i] for i,(seg,start) in enumerate(train.dataset.batches) if start<1024]
    row['train_subset']=evaluate(model,batches)
    print(width,'train_subset',row['train_subset'],flush=True)
    results[str(width)]=row
    (outdir/'checkpoint_probes.json').write_text(json.dumps(results,indent=2))
    del model
    torch.cuda.empty_cache()

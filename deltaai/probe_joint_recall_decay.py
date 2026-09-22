"""CPU-only reconstruction of first-layer scalar decay from saved GDN weights.

This is not a full model evaluation. It measures scalar retention on the path
from the matching information value position to the query key; it omits delta
transition products and information potentially encoded at neighboring positions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

p=argparse.ArgumentParser();p.add_argument('roots',type=Path,nargs='+');args=p.parse_args()
torch.set_num_threads(2);rows=[]
for root in args.roots:
    manifest=json.loads((root/'data/manifest.json').read_text())
    record=(root/'dataset_config.json').exists() and json.loads((root/'dataset_config.json').read_text()).get('layout')=='record'
    for d in [1,3]:
        folder=root/f'shared-gdn_current-d{d}-w64-s123';path=folder/'checkpoint.pt'
        if not path.exists():continue
        ck=torch.load(path,map_location='cpu',weights_only=False);state=ck['model']
        prefix='backbone.layers.0.mixer.gdn.'
        for cell in manifest['cells']:
            name=cell['name'];c=cell['contexts'];k=cell['keys']
            x=np.load(root/'data'/f'{name}-validation-shared.npy',mmap_mode='r')[:128].astype(np.int64)
            pos=np.load(root/'data'/f'{name}-validation-positions.npy',mmap_mode='r')[:128].astype(np.int64)
            b,L=x.shape
            if record:
                ip=2+3*np.arange(c*k);ictx=x[:,ip-1];qctx=np.take_along_axis(x,pos-1,1)
            else:
                ih=1+np.arange(c)*(1+2*k);qh=L//2+1+np.arange(c)*(1+2*k)
                ip=(ih[:,None]+1+2*np.arange(k)).reshape(-1)
                ictx=np.repeat(x[:,ih],k,axis=1);qctx=np.repeat(x[:,qh],k,axis=1)
            qkey=np.take_along_axis(x,pos,1)
            match=(qctx[:,:,None]==ictx[:,None,:])&(qkey[:,:,None]==x[:,None,ip])
            assert (match.sum(-1)==1).all();target=ip[match.argmax(-1)]+1
            with torch.no_grad():
                embed=F.embedding(torch.from_numpy(x),state['backbone.embeddings.word_embeddings.weight'])
                u=embed*torch.rsqrt(embed.square().mean(-1,keepdim=True)+1e-5)*state['backbone.layers.0.norm.weight']
                raw=F.linear(u,state[prefix+'a_proj.weight'])
                logs=-state[prefix+'A_log'].exp()*F.softplus(raw+state[prefix+'dt_bias']);s=logs.cumsum(1)
                h=s.shape[-1]
                retention=s.gather(1,torch.from_numpy(pos)[:,:,None].expand(-1,-1,h))-s.gather(1,torch.from_numpy(target)[:,:,None].expand(-1,-1,h))
                row=dict(trial=root.name,d=d,epoch=ck['epoch'],cell=name,
                    mean_alpha=logs.exp().mean((0,1)).tolist(),
                    mean_target_retention=retention.exp().mean((0,1)).tolist(),
                    median_target_log_retention=retention.flatten(0,1).median(0).values.tolist())
                rows.append(row);print(json.dumps(row),flush=True)
out=Path(__file__).resolve().parent/'results/joint_recall_iterations/first_layer_decay_probe.json'
out.write_text(json.dumps(dict(method=__doc__,rows=rows),indent=2)+'\n')

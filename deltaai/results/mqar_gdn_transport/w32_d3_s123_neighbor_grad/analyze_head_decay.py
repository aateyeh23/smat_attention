"""Read-only first-layer gate analysis from saved weights and cached examples.

No model construction, recurrence, backward, optimizer, or GPU job. Reconstruct
embedding -> RMSNorm -> forget gate exactly as the first block specifies.
"""
from pathlib import Path
import json
import torch
from torch.nn import functional as F

torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent
DELTA=OUT.parents[2]
CACHE=DELTA/'results/mqar_gdn_four_reads/modal_bundle/zoology_cache'
FILES=['data_25f7a1e8f8e6d2ba1fdbfa481d66614a.pt',
       'data_5e0c59f2a5df62cc996f523c97b0d10a.pt',
       'data_9892a64887c1ed56c1c7421f492d76ed.pt',
       'data_b53959d96a217faa44c26a74b7d53685.pt',
       'data_2e72f812c43c426a106c777f791c5e7d.pt']
CHECKPOINTS=[('neighbor_epoch3',OUT/'probe-epoch3.pt'),
             ('neighbor_epoch6',OUT/'w32-d3.pt'),
             ('original_transport_epoch23',OUT.with_name('w32_d3_s123')/'w32-d3.pt'),
             ('gdn_baseline_epoch32',DELTA/'results/mqar_width_sweeps/gdn/w32-d1.pt'),
             ('original_smat_epoch32',DELTA/'results/mqar_width_sweeps/gdn/w32-d3.pt')]
rows=[]
with torch.no_grad():
    for name,path in CHECKPOINTS:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        state=ck['model'];prefix='backbone.layers.0.mixer.'
        for file in FILES:
            data=torch.load(CACHE/file,map_location='cpu',weights_only=False)
            x,y=data['inputs'][:128],data['labels'][:128]
            pairs=data['slices']['num_kv_pairs'];n=x.shape[1]//2
            embed=F.embedding(x,state['backbone.embeddings.word_embeddings.weight'])
            u=embed*torch.rsqrt(embed.square().mean(-1,keepdim=True)+1e-5)
            u=u*state['backbone.layers.0.norm.weight']
            raw=F.linear(u,state[prefix+'gdn.a_proj.weight'])
            log_alpha=-state[prefix+'gdn.A_log'].exp()*F.softplus(raw+state[prefix+'gdn.dt_bias'])
            query_mask=y[:,n:]!=-100
            match=(x[:,n:,None]==x[:,None,:2*pairs:2]).long().argmax(-1)
            target_position=2*match+1
            cumulative=log_alpha.cumsum(1)
            written=cumulative.gather(1,target_position[:,:,None].expand(-1,-1,2))
            path_logs=(cumulative[:,n:]-written)[query_mask]
            beta=F.linear(u,state[prefix+'gdn.b_proj.weight']).sigmoid()
            row=dict(checkpoint=name,epoch=ck['next_epoch'],pairs=pairs,length=x.shape[1],
                     alpha_mean_per_head=log_alpha.exp().mean((0,1)).tolist(),
                     query_path_decay_mean_per_head=path_logs.exp().mean(0).tolist(),
                     query_path_log_decay_quantiles=torch.quantile(path_logs,
                         torch.tensor([.1,.5,.9]),dim=0).tolist(),
                     fraction_query_paths_below_1e_minus6=(path_logs<torch.log(torch.tensor(1e-6))).float().mean(0).tolist(),
                     beta_mean_per_head=beta.mean((0,1)).tolist())
            rows.append(row)
            print(json.dumps(row),flush=True)
(OUT/'head-decay-analysis.json').write_text(json.dumps(dict(method=__doc__,rows=rows),indent=2)+'\n')

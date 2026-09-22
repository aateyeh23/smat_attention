"""GPU diagnostic of stored-value visibility, transport and memory gates.

Value-position scores omit payload information carried at neighboring positions;
these are mechanism diagnostics, not a decomposition of prediction accuracy.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.nn import functional as F

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--folder',type=Path,required=True);p.add_argument('--examples',type=int,default=64)
p.add_argument('--initialization',action='store_true',help='Probe the exact seeded initialization instead of the checkpoint')
p.add_argument('--disable-scalar-decay',action='store_true',help='Counterfactual forward only; leave all checkpoint weights unchanged')
p.add_argument('--output',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.source/'deltaai'));sys.path.insert(0,str(args.source/'smat'))
import joint_recall as jr

root=args.folder.parent;recipe=json.loads((args.folder/'recipe.json').read_text());jr.configure_dataset(root)
manifest=json.loads((root/'data/manifest.json').read_text());cell=max(manifest['cells'],key=lambda c:c['records'])
c,k=cell['contexts'],cell['keys'];name=cell['name'];condition=recipe['condition']
x=np.load(root/'data'/f'{name}-validation-{condition}.npy',mmap_mode='r')[:args.examples].astype('int64')
positions=np.load(root/'data'/f'{name}-validation-positions.npy',mmap_mode='r')[:args.examples].astype('int64')
targets=np.load(root/'data'/f'{name}-validation-targets.npy',mmap_mode='r')[:args.examples].astype('int64')
b,L=x.shape;n=L//2
if jr.LAYOUT=='record':
    ip=2+3*np.arange(c*k);ictx=x[:,ip-1];qctx=np.take_along_axis(x,positions-1,1)
else:
    ih=1+np.arange(c)*(1+2*k);qh=n+1+np.arange(c)*(1+2*k)
    ip=(ih[:,None]+1+2*np.arange(k)).reshape(-1)
    ictx=np.repeat(x[:,ih],k,axis=1);qctx=np.repeat(x[:,qh],k,axis=1)
match=(qctx[:,:,None]==ictx[:,None,:])&(np.take_along_axis(x,positions,1)[:,:,None]==x[:,None,ip])
assert (match.sum(-1)==1).all()
target_index=torch.from_numpy(match.argmax(-1)).cuda()
value_positions=torch.from_numpy(ip+1).cuda();query_positions=torch.from_numpy(positions[0]).cuda()
torch.set_num_threads(2)
random.seed(recipe['seed']);np.random.seed(recipe['seed']);torch.manual_seed(recipe['seed']);torch.cuda.manual_seed_all(recipe['seed'])
model=jr.make_model(recipe['family'],recipe['width'],recipe['d'],recipe['lengths'],jr.VOCAB)
if recipe.get('gdn_memory_scalar_decay') is False:
    if not hasattr(jr,'apply_model_options'):raise ValueError('Source cannot reconstruct memory option')
    jr.apply_model_options(model,recipe)
if args.initialization:
    tau=recipe.get('memory_timescale_initialization',0)
    if tau:
        with torch.no_grad():
            for module in model.modules():
                if module.__class__.__name__=='GatedDeltaNet':
                    module.dt_bias.copy_(torch.log(torch.expm1(1/(tau*module.A_log.float().exp()))))
    ck=dict(epoch=0,steps=0,module_steps={})
else:
    ck=torch.load(args.folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'])
model.eval()
if args.disable_scalar_decay:
    assert recipe['family']=='gdn_current' and recipe['d']>=2
    for module in model.modules():
        if hasattr(module,'transport_scalar_decay'):module.transport_scalar_decay=False
for module_name,module in model.named_modules():
    if hasattr(module,'_steps'):module._steps=ck['module_steps'].get(module_name,0)
rows=[];handles=[]
def pre_hook(module_name):
    def hook(module,inputs):
        u=inputs[0].float();g=module.gdn
        raw=F.linear(u,g.a_proj.weight.float())
        logs=-g.A_log.float().exp()*F.softplus(raw+g.dt_bias.float())
        beta=F.linear(u,g.b_proj.weight.float()).sigmoid()*(2 if g.allow_neg_eigval else 1)
        cumulative=logs.cumsum(1);h=logs.shape[-1]
        stored=value_positions[target_index]
        retention=cumulative[:,query_positions]-cumulative.gather(1,stored[:,:,None].expand(-1,-1,h))
        row=dict(module=module_name,kind='gates',mean_alpha=logs.exp().mean((0,1)).tolist(),
                 median_log_scalar_retention=retention.flatten(0,1).median(0).values.tolist(),
                 beta_information_values=beta[:,value_positions].mean((0,1)).tolist(),
                 beta_queries=beta[:,n:].mean((0,1)).tolist())
        if hasattr(module,'alpha'):
            lam=torch.sigmoid(module.alpha.float()+F.linear(u[:,query_positions],module.lam_w.weight.float(),module.lam_w.bias.float()))
            row['memory_gate']=lam.mean((0,1)).tolist()
        if hasattr(module,'transport_scalar_decay'):
            row['extra_memory_scalar_decay']=module.transport_scalar_decay
        rows.append(row)
    return hook

def memory_hook(module_name):
    def hook(ca,inputs,output):
        _,phi,psi,values,boundary=inputs[:5];h=ca.h
        qi=query_positions-boundary
        q=phi[:,query_positions].float();keys=psi[:,value_positions].float()
        scores=torch.einsum('bqr,bjr->bqj',q,keys)
        target=target_index.repeat_interleave(h,0)
        cells=ca.last_write_idx[:,value_positions,0]
        selected=ca.last_read_idx[:,qi];weights=ca.last_read_weights[:,qi]
        if ca.mode=='plane':membership=ca.M.flatten(0,1)[selected[:,:,:,None],cells[:,None,None,:]]
        else:membership=(selected[:,:,:,None]==cells[:,None,None,:]).float()
        geometry=(membership*weights[:,:,:,None]).sum(2)
        target_geometry=geometry.gather(2,target[:,:,None]).squeeze(-1)
        target_score=scores.gather(2,target[:,:,None]).squeeze(-1)
        combined=scores*geometry
        qnorm=q.norm(dim=-1);knorm=keys.norm(dim=-1)
        row=dict(module=module_name,kind='memory',heads=h,buckets=ca.N0,q=ca.q,mode=ca.mode,
            target_coverage=float((target_geometry>0).float().mean()),
            uniform_value_coverage=float((geometry>0).float().mean()),
            target_geometric_share=float((target_geometry/geometry.sum(-1).clamp_min(1e-30)).mean()),
            uniform_share=1/(c*k),target_content_top1=float((scores.argmax(-1)==target).float().mean()),
            target_absolute_content_top1=float((scores.abs().argmax(-1)==target).float().mean()),
            target_absolute_combined_top1=float((combined.abs().argmax(-1)==target).float().mean()),
            median_absolute_target_score=float(target_score.abs().median()),
            median_query_norm_by_quartile=[float(t.median()) for t in qnorm.chunk(4,dim=1)],
            median_key_norm_by_quartile=[float(t.median()) for t in knorm.chunk(4,dim=1)],
            memory_output_rms=float(output.float().square().mean().sqrt()))
        rows.append(row)
    return hook

for module_name,module in model.named_modules():
    if hasattr(module,'gdn'):handles.append(module.register_forward_pre_hook(pre_hook(module_name)))
    if module.__class__.__name__=='ContentAssign':handles.append(module.register_forward_hook(memory_hook(module_name)))
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    logits,y=jr.query_logits(model,(x,positions,targets));accuracy=float((logits.argmax(-1)==y).float().mean())
for handle in handles:handle.remove()
out=dict(folder=str(args.folder),epoch=ck['epoch'],steps=ck['steps'],cell=name,examples=b,
         initialization=args.initialization,counterfactual_disable_scalar_decay=args.disable_scalar_decay,
         accuracy=accuracy,method=__doc__,rows=rows)
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

"""Bounded evaluation-only checkpoint ablations; no optimizer or training updates."""
import os
import sys
sys.path[:0]=['/opt/transport','/opt/smat/deltaai','/opt/smat/smat','/opt/zoology','/opt/fla-source']
os.environ.update(MQAR_FAMILY='gdn',ZOO_DM='32',ZOO_DS='3',ZOO_EPOCHS='32')
os.environ.pop('ZOO_SMOKE',None)
import json
from collections import defaultdict
from pathlib import Path
import torch
from zoology.model import LanguageModel
from zoology.data.utils import prepare_data
from zoo_gdn_transport_configs import configs
import zoo_gdn_transport as architecture
from zoo_gdn_transport import SmatGDNTransport

torch.set_num_threads(2)
torch.manual_seed(123)
ck=torch.load('/transport-results/seed123/neighbor_write_grad/w32-d3.pt',map_location='cpu',weights_only=False)
model=LanguageModel(configs[0].model).cuda().eval()
model.load_state_dict(ck['model'],strict=True)
_,test=prepare_data(configs[0].data)
batches=[]
for segment in test.dataset.segments:
    for start in range(0,128,32):
        batches.append((segment.inputs[start:start+32],segment.labels[start:start+32],segment.slices))
mixers=[m for m in model.modules() if isinstance(m,SmatGDNTransport)]
context={}
stats=defaultdict(lambda:defaultdict(float))
original_transport=architecture.boundary_transport


def add(key,name,values):
    stats[key][name+'_sum']+=values.double().sum().item()
    stats[key][name+'_count']+=values.numel()


def indices():
    x,y=context['x'],context['y'];n=x.shape[1]//2;pairs=context['pairs']
    mask=y[:,n:]!=-100
    match=(x[:,n:,None]==x[:,None,:2*pairs:2]).long().argmax(-1)
    return mask,2*match+1


def transport(q,k,beta,g,n):
    mode=context['mode']
    if mode=='no_transport':
        result=(k[:,:n]*beta[:,:n,...,None],q[:,n:])
    elif mode=='no_delta_transport':
        far=g[:,:n].cumsum(1)
        result=(k[:,:n]*(beta[:,:n]*(far[:,-1:]-far).exp())[...,None],
                q[:,n:]*g[:,n:].cumsum(1).exp()[...,None])
    else:
        result=original_transport(q,k,beta,torch.zeros_like(g) if mode=='no_decay' else g,n)
    if mode=='normal':
        layer=context['layer'];key=f'{context["pairs"]}/layer{layer}'
        mask,pos=indices();b,_,h,r=q.shape
        gather=pos[:, :,None,None].expand(-1,-1,h,r)
        fk=result[0].gather(1,gather)
        original_k=(beta[:,:n,...,None]*k[:,:n]).gather(1,gather)
        cumulative=g.cumsum(1)
        write_cum=cumulative[:,:n].gather(1,pos[:,:,None].expand(-1,-1,h))
        decay=(cumulative[:,n:]-write_cum).exp()
        add(key,'alpha',g.exp())
        add(key,'path_scalar_decay',decay[mask])
        add(key,'far_key_norm_ratio',(fk.norm(dim=-1)/original_k.norm(dim=-1).clamp_min(1e-12))[mask])
        add(key,'recent_query_norm_ratio',(result[1].norm(dim=-1)/q[:,n:].norm(dim=-1).clamp_min(1e-12))[mask])
    return result


architecture.boundary_transport=transport


def install(m,layer):
    m.register_forward_pre_hook(lambda module,args:context.update(layer=layer))
    captured={}
    for mods in m.ca.values():
        for ca in mods.values():
            read_original=ca._topk_reads
            forward_original=ca.forward
            def read(u,ca=ca,read_original=read_original):
                idx,w=read_original(u)
                if context['mode']!='oracle_reads': return idx,w
                mask,pos=indices();b,n=pos.shape;h=ca.h
                target=ca.last_hard_k.reshape(b,h,n).gather(2,pos[:,None].expand(-1,h,-1))
                membership=ca.M.flatten(0,1).T.bool()[target]
                scores=torch.einsum('blm,hem->bhle',ca.ln(u),ca.W_read)+ca.b_read[None,:,None,:]
                forced=(scores+membership*1e6).topk(4,-1).indices
                fw=scores.gather(-1,forced).masked_fill(~membership.gather(-1,forced),-torch.inf).softmax(-1)
                supervised=mask[:,None,:,None]
                return (torch.where(supervised,forced,idx.reshape(b,h,n,4)).flatten(0,1),
                        torch.where(supervised,fw,w.reshape(b,h,n,4)).flatten(0,1))
            def forward(*args,ca=ca,forward_original=forward_original,**kwargs):
                out=forward_original(*args,**kwargs)
                if context['mode']=='normal':
                    mask,pos=indices();b,n=pos.shape;h=ca.h
                    key=f'{context["pairs"]}/layer{layer}'
                    target=ca.last_write_idx[...,0].reshape(b,h,n).gather(2,pos[:,None].expand(-1,h,-1))
                    reads=ca.last_read_idx.reshape(b,h,n,4)
                    hits=ca.M.flatten(0,1).bool()[reads,target[...,None]]
                    weights=ca.last_read_weights.reshape(b,h,n,4)
                    hm=mask[:,None].expand(-1,h,-1)
                    add(key,'coverage_per_head',hits.any(-1)[hm])
                    add(key,'coverage_any_head',hits.any(-1).any(1)[mask])
                    add(key,'weight_on_target',(hits*weights).sum(-1)[hm])
                    # Coverage alone can be high if reads include nearly every
                    # value. Measure selectivity and write collisions separately.
                    pairs=context['pairs']
                    value_buckets=ca.last_write_idx[...,0].reshape(b,h,n)[:,:,1:2*pairs:2]
                    all_hits=ca.M.flatten(0,1).bool()[reads[...,None],value_buckets[:,:,None,None,:]]
                    coefficients=(all_hits*weights[...,None]).sum(3)
                    accessible=all_hits.any(3).sum(-1)
                    add(key,'accessible_values',accessible[hm])
                    target_weight=(hits*weights).sum(-1)
                    add(key,'target_fraction_of_value_weight',(target_weight/coefficients.sum(-1).clamp_min(1e-12))[hm])
                    equal=value_buckets[...,None]==value_buckets[...,None,:]
                    unique=(~torch.tril(equal,diagonal=-1).any(-1)).sum(-1)
                    add(key,'occupied_value_buckets',unique)
                    add(key,'value_bucket_collisions',equal.sum(-1)-1)
                    # Is the temporal content score itself aligned to the key's
                    # corresponding value before profile aggregation?
                    mq=args[1][:,n:].reshape(b,h,n,-1)
                    mk=args[2][:,:n].reshape(b,h,n,-1)[:,:,1:2*pairs:2]
                    scores=torch.einsum('bhir,bhjr->bhij',mq,mk)
                    match=(pos-1)//2
                    expected=match[:,None].expand(-1,h,-1)
                    add(key,'correct_content_top1',(scores.argmax(-1)==expected)[hm])
                    add(key,'correct_absolute_content_top1',(scores.abs().argmax(-1)==expected)[hm])
                    routed=scores*coefficients
                    add(key,'correct_routed_absolute_top1',(routed.abs().argmax(-1)==expected)[hm])
                    add(key,'n_cells',unique.new_full(unique.shape,ca.N0))
                    u=args[0]
                    lam=torch.sigmoid(m.alpha+m.lam_w(u.float())[:,n:])
                    contribution=out.reshape(b,h,n,-1).permute(0,2,1,3)*lam[...,None]
                    captured.update(contribution=contribution,mask=mask,key=key,n=n)
                    add(key,'gate',lam[mask])
                return out
            ca._topk_reads=read;ca.forward=forward
    def before_norm(module,args):
        if context['mode']!='normal': return
        c=captured;full=args[0][:,c['n']:];local=full-c['contribution'];mask=c['mask']
        add(c['key'],'memory_squared',c['contribution'][mask].square())
        add(c['key'],'local_squared',local[mask].square())
    m.gdn.o_norm.register_forward_pre_hook(before_norm)


for i,m in enumerate(mixers): install(m,i)
results={'checkpoint_epoch':ck['next_epoch'],'saved_full_metrics':ck['metrics'],
         'examples_per_cell':128,'interventions':{}}
with torch.no_grad():
    for mode in (('normal',) if os.environ.get('PROBE_ONLY_NORMAL')=='1' else
                 ('normal','memory_off','no_decay','no_delta_transport','no_transport','oracle_reads')):
        context['mode']=mode
        for m in mixers: m.disable_g=mode=='memory_off'
        counts=defaultdict(lambda:defaultdict(float))
        for x,y,slices in batches:
            x,y=x.cuda(),y.cuda();pairs=slices['num_kv_pairs'];n=x.shape[1]//2
            context.update(x=x,y=y,pairs=pairs)
            pred=model(x).argmax(-1);mask=y!=-100;correct=(pred==y)&mask
            row=counts[pairs]
            row['sum']+=(correct.sum(-1)/mask.sum(-1)).sum().item();row['examples']+=len(x)
            for label,sl in [('first',slice(None,n)),('second',slice(n,None))]:
                row[label+'_correct']+=correct[:,sl].sum().item();row[label+'_count']+=mask[:,sl].sum().item()
        metrics={str(k):{'accuracy':v['sum']/v['examples'],**{label+'_accuracy':v[label+'_correct']/v[label+'_count'] if v[label+'_count'] else None for label in ('first','second')}} for k,v in counts.items()}
        results['interventions'][mode]=metrics
        print('INTERVENTION '+json.dumps(dict(mode=mode,epoch=ck['next_epoch'],metrics=metrics)),flush=True)
results['internal_stats']={}
for key,v in stats.items():
    results['internal_stats'][key]={name[:-4]:value/v[name[:-4]+'_count'] for name,value in v.items() if name.endswith('_sum')}
    row=results['internal_stats'][key]
    row['memory_to_local_rms']=(row['memory_squared']/max(row['local_squared'],1e-30))**.5
print('PROBE_RESULT '+json.dumps(results),flush=True)

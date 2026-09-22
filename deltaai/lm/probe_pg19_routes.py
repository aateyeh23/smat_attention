"""Read-only routing and memory-contribution probe on frozen PG19 checkpoints."""
import argparse, hashlib, importlib, inspect, json, math, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def stats(x):
    x=x.detach().float()
    return dict(mean=x.mean().item(), rms=x.square().mean().sqrt().item(),
                abs_max=x.abs().max().item(), zero_fraction=(x==0).float().mean().item())


def occupancy(ids, size):
    counts=torch.zeros(ids.shape[0],size,device=ids.device)
    counts.scatter_add_(1,ids.reshape(ids.shape[0],-1).long(),torch.ones_like(ids.reshape(ids.shape[0],-1),dtype=torch.float))
    p=counts/counts.sum(-1,keepdim=True)
    return dict(used_per_head=(counts>0).sum(-1).tolist(), total=size,
                max_bucket_fraction=p.max(-1).values.tolist(),
                normalized_entropy=(-(p*p.clamp_min(1e-30).log()).sum(-1)/math.log(size)).tolist())


def main(args):
    torch.set_num_threads(4)
    source=Path(args.checkpoint)
    manifest=json.loads((source.parent/'manifest.json').read_text())
    for f,h in manifest['source_sha256'].items():
        if hashlib.sha256(Path('/opt/pg19-rank64',f).read_bytes()).hexdigest()!=h:
            raise ValueError('Source mismatch '+f)
    # Open once: live training replaces latest.pt atomically, so this is one snapshot.
    with source.open('rb') as fp:
        digest=hashlib.file_digest(fp,'sha256').hexdigest();fp.seek(0)
        saved=torch.load(fp,map_location='cpu',weights_only=False)
    family=manifest['family']
    if manifest.get('kernel_optimization')=='tiled_fp32_v1':
        import pg19_optimized_kernels
        pg19_optimized_kernels.enable()
    module=importlib.import_module('gdn_smat_transport_scale' if family=='gdn' else 'mamba_smat_scale')
    cfg=module.ScaleConfig(**saved['config']);model=module.ScaleLM(cfg).cuda().eval()
    model.load_state_dict(saved['model'],strict=True)
    rows=saved['recipe']['eval_rows'][:2]
    result=dict(arm=source.parent.name,checkpoint=str(source),checkpoint_sha256=digest,
                tokens=saved['cursor']*cfg.length,rows=rows,layers=[])
    del saved
    data=np.memmap('/data/pg19-16k-2b/val.bin',mode='r',dtype=np.uint16).reshape(-1,cfg.length+1)
    raw=torch.from_numpy(np.asarray(data[rows],dtype=np.int64)).cuda();x,y=raw[:,:-1],raw[:,1:]
    cas=[];handles=[];by_id={}
    for i,block in enumerate(model.blocks):
        mixer=block.mixer;record=dict(layer=i);result['layers'].append(record);by_id[id(mixer)]=record
        ca=next(iter(mixer.ca[str(cfg.length)].values()));cas.append(ca)
        def hook(c,args,out,record=record):
            record['raw_read']=stats(out)
            record['writes']=occupancy(c.last_write_idx,c.N0)
            record['reads']=occupancy(c.last_read_idx,c.W_read.shape[1])
            record['read_weights']=stats(c.last_read_weights)
            record['weight_sum_max_error']=(c.last_read_weights.sum(-1)-1).abs().max().item()
            record['write_weights']=stats(c.last_write_weights)
        handles.append(ca.register_forward_hook(hook))
    codes={}
    for block in model.blocks:
        fun=block.mixer.forward.__func__;lines,start=inspect.getsourcelines(fun)
        targets=[start+j for j,l in enumerate(lines) if 'o = torch.cat((o[:,:n]' in l or 'y = torch.cat([y[:, :n], y[:, n:] +' in l]
        if len(targets)!=1:raise ValueError('Cannot locate memory injection')
        codes[fun.__code__]=targets[0]
    def trace(frame,event,arg):
        if frame.f_code not in codes:return None
        if event=='line' and frame.f_lineno==codes[frame.f_code]:
            v=frame.f_locals;r=by_id[id(v['self'])];n=v['n']
            local=v['o'][:,n:] if family=='gdn' else v['y'][:,n:]
            memory=v['out'] if family=='gdn' else v['Ylr']
            delta=v['lam'].unsqueeze(-1)*memory
            actual=(local+delta.to(local.dtype)).float()-local.float()
            r['gate']=stats(v['lam']);r['local']=stats(local);r['injected']=stats(delta);r['actual_change']=stats(actual)
            r['relative_rms']=r['injected']['rms']/max(r['local']['rms'],1e-30)
            r['position_bins']=[]
            for lo,hi in [(0,128),(128,1024),(1024,4096),(4096,8192)]:
                q=stats(delta[:,lo:hi]);q.update(offset_start=lo,offset_end=hi,
                    relative_rms=q['rms']/max(stats(local[:,lo:hi])['rms'],1e-30))
                r['position_bins'].append(q)
            if 'qry_w' in v:r['query_decay']=stats(v['qry_w'])
        return trace
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        sys.settrace(trace)
        try:normal,_=model.hidden(x)
        finally:sys.settrace(None)
        for h in handles:h.remove()
        handles=[c.register_forward_hook(lambda m,a,o:torch.zeros_like(o)) for c in cas]
        try:ablated,_=model.hidden(x)
        finally:
            for h in handles:h.remove()
        total_loss=[0.,0.];changed=0;maxdiff=0.;kl=0.;n=cfg.length//2
        for start in range(n,cfg.length,512):
            stop=min(start+512,cfg.length);target=y[:,start:stop].flatten()
            a=F.linear(normal[:,start:stop],model.embedding.weight).float().flatten(0,1)
            b=F.linear(ablated[:,start:stop],model.embedding.weight).float().flatten(0,1)
            for j,z in enumerate([a,b]):total_loss[j]+=F.cross_entropy(z,target,reduction='sum').item()
            changed+=(a.argmax(-1)!=b.argmax(-1)).sum().item();maxdiff=max(maxdiff,(a-b).abs().max().item())
            la=a.log_softmax(-1);lb=b.log_softmax(-1);kl+=(la.exp()*(la-lb)).sum().item()
        count=len(rows)*n
        result['ablation']=dict(second_half_tokens=count,normal_loss=total_loss[0]/count,
            zero_memory_loss=total_loss[1]/count,loss_increase=(total_loss[1]-total_loss[0])/count,
            top1_changed_fraction=changed/count,max_logit_difference=maxdiff,mean_kl=kl/count)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())

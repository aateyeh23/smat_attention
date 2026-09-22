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
    cas=[next(iter(block.mixer.ca[str(cfg.length)].values())) for block in model.blocks]
    result.pop('layers')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        normal,_=model.hidden(x)
        handles=[c.register_forward_hook(lambda m,a,o:torch.zeros_like(o)) for c in cas]
        try:ablated,_=model.hidden(x)
        finally:
            for h in handles:h.remove()
        result['regions']={}
        for label,n in [('second_half',cfg.length//2),('last_2k',cfg.length-2048)]:
            total_loss=[0.,0.];changed=0;kl=0.
            per_row=torch.zeros(2,len(rows),device='cuda',dtype=torch.float64)
            for start in range(n,cfg.length,512):
                stop=min(start+512,cfg.length);target=y[:,start:stop].flatten()
                a=F.linear(normal[:,start:stop],model.embedding.weight).float().flatten(0,1)
                b=F.linear(ablated[:,start:stop],model.embedding.weight).float().flatten(0,1)
                for j,z in enumerate([a,b]):
                    losses=F.cross_entropy(z,target,reduction='none').reshape(len(rows),-1)
                    per_row[j]+=losses.double().sum(-1)
                changed+=(a.argmax(-1)!=b.argmax(-1)).sum().item()
                la=a.log_softmax(-1);lb=b.log_softmax(-1);kl+=(la.exp()*(la-lb)).sum().item()
            count=len(rows)*(cfg.length-n)
            total_loss=per_row.sum(-1).tolist()
            result['regions'][label]=dict(tokens=count,normal_loss=total_loss[0]/count,
                zero_memory_loss=total_loss[1]/count,loss_increase=(total_loss[1]-total_loss[0])/count,
                per_sequence_loss_increase=((per_row[1]-per_row[0])/(cfg.length-n)).tolist(),
                top1_changed_fraction=changed/count,mean_kl=kl/count)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())

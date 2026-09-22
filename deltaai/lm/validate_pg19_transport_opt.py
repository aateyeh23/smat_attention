"""Compare optimized transport on trained 16-layer checkpoints and real PG19."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import smat_gdn_transport as transport
import pg19_optimized_kernels
from gdn_smat_transport_scale import ScaleConfig, ScaleLM


def emit(event, **data):
    print(json.dumps(dict(event=event, **data)),flush=True)


def main(output, chunk):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    original_scan=transport.chunk_gated_delta_rule
    # Snapshot immutable checkpoint files before comparison. Active runs continue.
    for d in (2,3):
        root=out/f'd{d}'; root.mkdir(exist_ok=True)
        source=Path('/checkpoints/seed123/six-500m-20260917')/f'gdn-smat-d{d}'
        shutil.copy2(source/'latest.pt',root/'latest.pt')
        shutil.copy2(source/'recipe.json',root/'recipe.json')
        saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
        cfg=ScaleConfig(**saved['config'])
        torch.manual_seed(cfg.seed)
        model=ScaleLM(cfg).cuda().train()
        model.load_state_dict(saved['model'])
        params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        data=np.memmap('/data/pg19-16k-2b/train.bin',mode='r',dtype=np.uint16).reshape(-1,cfg.length+1)
        order=np.random.default_rng(cfg.seed).permutation(len(data))
        ids=order[saved['cursor']:saved['cursor']+2]
        raw=torch.from_numpy(np.asarray(data[ids],dtype=np.int64)).cuda()
        x,y=raw[:,:-1],raw[:,1:]
        emit('snapshot',d=d,step=saved['step'],tokens=saved['cursor']*cfg.length)
        reference=None
        for optimized,current_chunk in ((False,16),(True,chunk)):
            if optimized:
                pg19_optimized_kernels.enable()
            else:
                pg19_optimized_kernels.disable()
            def scan(*args,**kwargs):
                kwargs['chunk_size']=current_chunk
                return original_scan(*args,**kwargs)
            transport.chunk_gated_delta_rule=scan
            def compute():
                model.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,aux=model.loss(x,y)
                (loss+aux).backward()
                return loss.item(),aux.item()
            values=compute()
            grads={n:p.grad.detach().cpu().clone() for n,p in params}
            assert all(t.isfinite().all() for t in grads.values())
            if reference is None:
                reference=(values,grads)
            else:
                differences=[]; total_diff=0.;total_ref=0.
                for n,g in grads.items():
                    ref=reference[1][n]; delta=g-ref
                    diff=delta.double().square().sum().item(); norm=ref.double().square().sum().item()
                    total_diff+=diff;total_ref+=norm
                    differences.append(dict(name=n,max_abs=delta.abs().max().item(),relative_l2=(diff/max(norm,1e-30))**.5))
                loss_diff=abs(values[0]-reference[0][0])
                rel=(total_diff/total_ref)**.5
                emit('trained_model_comparison',d=d,chunk=chunk,loss=values[0],reference_loss=reference[0][0],
                     loss_difference=loss_diff,aux=values[1],reference_aux=reference[0][1],gradient_relative_l2=rel,
                     worst_gradients=sorted(differences,key=lambda r:r['relative_l2'],reverse=True)[:12])
                assert loss_diff<2e-4 and rel<.005, 'Changed trained-model loss or gradients'
            times=[]
            for _ in range(3):
                torch.cuda.synchronize();start=time.perf_counter();compute();torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
            emit('full_model_benchmark',d=d,optimized=optimized,chunk=current_chunk,seconds=float(np.median(times)),
                 tokens_per_second=2*cfg.length/float(np.median(times)),peak_gb=torch.cuda.max_memory_allocated()/1e9)
            del grads
        del model,params,reference,saved,raw,x,y
        gc.collect();torch.cuda.empty_cache()
        # Real optimizer continuation, same order/schedule/global batch as the six-arm campaign.
        def scan(*args,**kwargs):
            kwargs['chunk_size']=chunk
            return original_scan(*args,**kwargs)
        transport.chunk_gated_delta_rule=scan
        import train_pg19_scale
        saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
        step=saved['step']; del saved
        args=argparse.Namespace(d=d,data='/data/pg19-16k-2b',output=str(root),tokens=500_000_000,
            batch=2,accumulation=4,global_batch_rows=8,lr=.0003,warmup_tokens=40_000_000,
            eval_every=50_000_000,eval_rows=64,checkpoint_every=250_000_000,log_every=1,max_hours=11.8,
            backend='triton',layers=16,width=768,ffn_width=2048,head_dim=128,heads=6,read_rank=64,
            max_steps=step+5,length=16384,loss_chunk=4096,activation_checkpointing=False)
        train_pg19_scale.train(args)
        emit('optimizer_pilot_passed',d=d,resumed_step=step,final_step=step+5,chunk=chunk)
        gc.collect();torch.cuda.empty_cache()
    emit('validation_complete',chunk=chunk)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--chunk',type=int,required=True)
    args=p.parse_args();main(args.output,args.chunk)

"""Isolated, reproducible H100 profiling of the active PG19 GDN geometry."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.profiler import profile, ProfilerActivity, record_function

import smat_gdn_transport as transport
import zoo_gdn_transport
from gdn_smat_transport_scale import Block, ScaleConfig


def emit(event, **data):
    print(json.dumps(dict(event=event, **data)), flush=True)


def measure(fn, warmup=2, repeat=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(repeat):
        begin=time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-begin)*1000)
    return sorted(samples)[len(samples)//2]


def main(output, optimized=False):
    out=Path(output); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(123)
    emit('device', name=torch.cuda.get_device_name(), torch=torch.__version__)
    original_scan=transport.chunk_gated_delta_rule
    if optimized:
        import smat_gdn_tiled
        smat_gdn_tiled.enable()
        import smat_read_triton, smat_read_tiled
        smat_read_triton.read_topk=smat_read_tiled.read_topk
        emit('optimized_kernels_enabled')
    original_transport=zoo_gdn_transport.boundary_transport
    def annotated(*args, **kwargs):
        with record_function('BOUNDARY_TRANSPORT'):
            return original_transport(*args, **kwargs)
    zoo_gdn_transport.boundary_transport=annotated
    for d in (2,3):
        cfg=ScaleConfig(d=d,width=768,ffn_width=2048,heads=6,head_dim=128,
                        length=16384,read_rank=64,read_backend='triton',fused_norm=True,
                        activation_checkpointing=False)
        block=Block(cfg,0).cuda().train()
        x=torch.randn(2,16384,768,device='cuda',requires_grad=True)
        probe=torch.randn_like(x)/x.numel()**.5
        ca=next(iter(block.mixer.ca['16384'].values()))
        original_ca=ca.forward
        def annotated_ca(*args, **kwargs):
            with record_function('SMAT_MEMORY'):
                return original_ca(*args, **kwargs)
        ca.forward=annotated_ca
        def step():
            block.zero_grad(set_to_none=True); x.grad=None
            with torch.autocast('cuda',dtype=torch.bfloat16):
                y,aux=block(x)
                loss=(y*probe).sum()+aux
            loss.backward()
        outputs={}
        for chunk in (16,32,64):
            def scan(*args, **kwargs):
                kwargs['chunk_size']=chunk
                return original_scan(*args, **kwargs)
            transport.chunk_gated_delta_rule=scan
            ms=measure(step)
            outputs[chunk]=x.grad.detach().clone()
            emit('layer_benchmark',d=d,chunk=chunk,ms=ms,peak_gb=torch.cuda.max_memory_allocated()/1e9)
            if chunk==16:
                with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                    step(); torch.cuda.synchronize()
                (out/f'd{d}-profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=70))
                emit('profile',d=d,table=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=18))
                for e in prof.key_averages():
                    if e.key in ('BOUNDARY_TRANSPORT','SMAT_MEMORY'):
                        emit('region',d=d,name=e.key,cpu_us=e.cpu_time_total,device_us=e.device_time_total)
            else:
                diff=(outputs[chunk]-outputs[16])
                emit('layer_input_gradient_difference',d=d,chunk=chunk,max_abs=diff.abs().max().item(),relative_l2=(diff.norm()/outputs[16].norm()).item(),finite=bool(outputs[chunk].isfinite().all()))
        del block,x,probe,outputs,ca,original_ca
        gc.collect(); torch.cuda.empty_cache()
    transport.chunk_gated_delta_rule=original_scan
    # Long-sequence derivatives: weak, moderate, and strong decay.
    for decay in (.0001,.01,1.):
        torch.manual_seed(930)
        q,k=[F.normalize(torch.randn(2,16384,6,128,device='cuda'),dim=-1).requires_grad_() for _ in range(2)]
        beta=torch.rand(2,16384,6,device='cuda').requires_grad_()
        g=(-torch.rand_like(beta)*decay).requires_grad_()
        probes=[torch.randn_like(q[:,:8192]) for _ in range(2)]
        reference=None
        for chunk in (16,32,64):
            def scan(*args,**kwargs):
                kwargs['chunk_size']=chunk
                return original_scan(*args,**kwargs)
            transport.chunk_gated_delta_rule=scan
            def compute():
                ys=original_transport(q,k,beta,g,8192)
                grads=torch.autograd.grad(ys,(q,k,beta,g),probes)
                return ys,grads
            if optimized and reference is None:
                smat_gdn_tiled.disable()
                ys,grads=compute()
                reference=tuple(t.detach() for t in (*ys,*grads))
                smat_gdn_tiled.enable()
            ms=measure(compute)
            ys,grads=compute()
            current=tuple(t.detach() for t in (*ys,*grads))
            if reference is None: reference=current
            diffs=[dict(max_abs=(a-b).abs().max().item(),relative_l2=((a-b).norm()/b.norm().clamp_min(1e-20)).item(),finite=bool(a.isfinite().all())) for a,b in zip(current,reference)]
            if optimized and chunk==16:
                assert all(item['finite'] and item['relative_l2']<1e-5 for item in diffs)
            emit('transport_benchmark',decay=decay,chunk=chunk,ms=ms,differences=diffs)
        del q,k,beta,g,probes,ys,grads,current,reference
        gc.collect(); torch.cuda.empty_cache()
    emit('complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    parser.add_argument('--optimized',action='store_true')
    args=parser.parse_args();main(args.output,args.optimized)

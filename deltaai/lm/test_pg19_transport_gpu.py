"""Full-context GPU validation for the updated PG19 transport configuration."""
import argparse
import os
import json
import time
import numpy as np
import torch
if os.environ.get('PG19_MEMORY_VARIANT') == 'mamba2_fixed_write':
    from mamba_smat_scale import ScaleConfig, ScaleLM
else:
    from gdn_smat_transport_scale import ScaleConfig, ScaleLM


def main(d):
    torch.manual_seed(123)
    # Check finite backward and causal short-prefix inference at a bounded width.
    cfg=ScaleConfig(d=d,width=64,layers=2,ffn_width=128,vocab=257,length=256,
                    head_dim=16,read_backend='triton',fused_norm=True,activation_checkpointing=False)
    model=ScaleLM(cfg).cuda()
    tokens=torch.randint(cfg.vocab,(1,256),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss,aux=model.loss(tokens,tokens.roll(-1,1))
    if os.environ.get('PG19_MEMORY_VARIANT') == 'mamba2_fixed_write':
        # Task loss alone must reach write-hash parameters, not just balance aux.
        hashes=[ca.W for block in model.blocks for modules in block.mixer.ca.values() for ca in modules.values()]
        grads=torch.autograd.grad(loss,hashes,retain_graph=True)
        assert all(g.isfinite().all() for g in grads)
        assert any(g.abs().max()>0 for g in grads),'No task-loss gradient to write hash'
        print('PASS MAMBA TASK-LOSS WRITE-HASH GRADIENT',flush=True)
    (loss+aux).backward()
    for name,p in model.named_parameters():
        if p.requires_grad:assert p.grad is not None and p.grad.isfinite().all(),name
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=model.next_logits(tokens[:,:192])
        altered=tokens.clone();altered[:,192:]=torch.randint(cfg.vocab,(1,64),device='cuda')
        h,_=model.hidden(altered)
        actual=torch.nn.functional.linear(h[:,191],model.embedding.weight).float()
        torch.testing.assert_close(actual,expected,atol=.02,rtol=.02)
    del model
    torch.cuda.empty_cache()
    # Full-width/full-context two-layer test exercises64-dimensional transport
    # and the d-specific16K geometry before allocating the complete16-layer LM.
    cfg=ScaleConfig(d=d,layers=2,activation_checkpointing=True,loss_chunk=4096,
                    read_backend='triton',fused_norm=True)
    model=ScaleLM(cfg).cuda().train()
    data=np.memmap('/data/pg19-16k-2b/train.bin',mode='r',dtype=np.uint16)
    tokens=torch.from_numpy(np.asarray(data[:16385],dtype=np.int64)[None]).cuda()
    start=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,aux=model.loss(tokens[:,:-1],tokens[:,1:])
    (loss+aux).backward()
    for name,p in model.named_parameters():
        if p.requires_grad:assert p.grad is not None and p.grad.isfinite().all(),name
    torch.cuda.synchronize()
    print('PASS '+json.dumps(dict(d=d,length=16384,width=1024,head_dim=64,
        loss=loss.item(),aux=aux.item(),seconds=time.monotonic()-start,
        peak_gb=torch.cuda.max_memory_allocated()/1e9)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--d',type=int,required=True)
    main(p.parse_args().d)

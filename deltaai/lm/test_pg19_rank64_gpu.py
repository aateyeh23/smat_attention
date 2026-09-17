"""GPU gate for rank64 routing and fixed-write-gradient PG19 models."""
import argparse
import os
import torch
from torch.nn import functional as F
if os.environ['PG19_MEMORY_VARIANT']=='mamba2_fixed_write':
    from mamba_smat_scale import ScaleConfig, ScaleLM
else:
    from gdn_smat_transport_scale import ScaleConfig, ScaleLM


def check(d):
    torch.set_num_threads(4)
    torch.manual_seed(731)
    cfg=ScaleConfig(d=d,width=64,layers=2,ffn_width=128,vocab=257,length=256,
        heads=2,head_dim=16,read_rank=32,read_backend='triton',fused_norm=True,
        activation_checkpointing=False)
    model=ScaleLM(cfg).cuda()
    ca=next(iter(model.blocks[0].mixer.ca['256'].values()))
    x=torch.randn(2,32,64,device='cuda',requires_grad=True)
    params=(x,ca.read_proj.weight,ca.W_read,ca.b_read)
    # Compare fused top4 selection and gradients to explicit dense projection.
    idx,weights=ca._topk_reads(x)
    dense=torch.einsum('her,rm->hem',ca.W_read,ca.read_proj.weight)
    logits=torch.einsum('blm,hem->bhle',ca.ln(x),dense)+ca.b_read[None,:,None,:]
    values,indices=logits.flatten(0,1).topk(4,dim=-1)
    reference=values.softmax(-1)
    assert torch.equal(idx,indices)
    torch.testing.assert_close(weights,reference,atol=2e-6,rtol=2e-5)
    dy=torch.randn_like(weights)
    g1=torch.autograd.grad(weights,params,dy,retain_graph=True)
    g2=torch.autograd.grad(reference,params,dy)
    for a,b in zip(g1,g2):torch.testing.assert_close(a,b,atol=2e-5,rtol=5e-4)
    print('PASS RANKED READER FORWARD/BACKWARD',flush=True)
    tokens=torch.randint(cfg.vocab,(1,256),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,aux=model.loss(tokens,tokens.roll(-1,1))
    hashes=[c.W for block in model.blocks for mods in block.mixer.ca.values() for c in mods.values()]
    grads=torch.autograd.grad(loss,hashes,retain_graph=True)
    assert all(g.isfinite().all() for g in grads)
    assert any(g.abs().max()>0 for g in grads),'Task-loss write-hash gradients missing'
    (loss+aux).backward()
    for name,p in model.named_parameters():
        if p.requires_grad:assert p.grad is not None and p.grad.isfinite().all(),name
    print('PASS MODEL BACKWARD AND TASK-LOSS WRITE-HASH GRADIENT',flush=True)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=model.next_logits(tokens[:,:192])
        tokens[:,192:]=torch.randint(cfg.vocab,(1,64),device='cuda')
        hidden,_=model.hidden(tokens)
        actual=F.linear(hidden[:,191],model.embedding.weight).float()
        torch.testing.assert_close(actual,expected,atol=.02,rtol=.02)
    print('PASS CAUSAL INFERENCE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--d',type=int,required=True)
    check(p.parse_args().d)

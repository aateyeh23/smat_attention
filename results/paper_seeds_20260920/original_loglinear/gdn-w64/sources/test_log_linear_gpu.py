"""GPU correctness gate; no optimizer steps or extra training runs."""
import json
import os
import time
from pathlib import Path
import types
import torch
from zoo_log_linear import log_linear, level_indices, LogLinearMixer


def reference(q,k,v,g,lam,beta=None):
    outputs=[]
    for t in range(q.shape[1]):
        terms=[]
        for s in range(t+1):
            state = k[:,s,:, :,None] * v[:,s,:,None,:]
            if beta is not None:
                state = state * beta[:,s,:,None,None]
            for j in range(s+1,t+1):
                state = state * g[:,j,:,None,None].exp()
                if beta is not None:
                    kj=k[:,j]
                    state = state - beta[:,j,:,None,None] * kj[...,None] * (kj[...,None] * state).sum(-2,keepdim=True)
            level=(t ^ s).bit_length()
            terms.append((q[:,t,:,:,None]*state).sum(-2) * lam[:,t,:,level,None])
        outputs.append(torch.stack(terms).sum(0))
    return torch.stack(outputs,1)


def validate():
    assert torch.cuda.is_available()
    torch.manual_seed(321)
    torch.backends.cuda.matmul.allow_tf32=False
    src=Path(os.environ.get('LL_UPSTREAM_BASE','/opt/log-linear/upstream_base.py')).read_text()
    # Defer optional jaxtyping annotations without changing any reference math.
    src='from __future__ import annotations\n'+src.replace('from jaxtyping import Float, Int','')
    upstream=types.ModuleType('upstream_base')
    exec(compile(src,'upstream_base.py','exec'),upstream.__dict__)
    original_levels=upstream.make_levels_matrix
    upstream.make_levels_matrix=lambda **kw: original_levels(**dict(kw,cached_length=None))
    for t in (8,64,128,256):
        actual=level_indices(t,'cuda')
        expected=torch.tensor([[0 if i<=j else upstream.get_level_index_weak(i,j,2)
                                for j in range(t)] for i in range(t)],device='cuda')
        assert torch.equal(actual.tril(),expected)
    for family in ('mamba2','gdn'):
        q=torch.randn(2,8,2,4,device='cuda',dtype=torch.double)*.2
        k=torch.randn_like(q)*.2; v=torch.randn_like(q)
        g=-torch.rand(2,8,2,device='cuda',dtype=torch.double)*.3
        lam=torch.rand(2,8,2,4,device='cuda',dtype=torch.double)+.1
        beta=torch.rand_like(g) if family=='gdn' else None
        variables=[x.requires_grad_() for x in (q,k,v,g,lam)]
        if beta is not None: variables.append(beta.requires_grad_())
        actual=log_linear(q,k,v,g,lam,beta)
        expected=reference(q,k,v,g,lam,beta)
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-9)
        with torch.no_grad():
            if beta is None:
                official=upstream.hattention_materialized_v2(q,k,v,g,lam.log(),2,upstream.HType.WEAK)
            else:
                official=upstream.hattention_materialized_dplr_v2(q,k,v,beta,g,lam,2,upstream.HType.WEAK)
            torch.testing.assert_close(actual,official,atol=1e-10,rtol=1e-9)
        dy=torch.randn_like(actual)
        ga=torch.autograd.grad(actual,variables,dy)
        ge=torch.autograd.grad(expected,variables,dy)
        for a,e in zip(ga,ge): torch.testing.assert_close(a,e,atol=1e-9,rtol=1e-8)
        print('PASS reference forward/backward '+family,flush=True)
    for family in ('gdn','mamba2'):
        for width in (16,32,64):
            model=LogLinearMixer(width,family=family).cuda()
            for t in (64,128,256):
                x=torch.randn(2,t,width,device='cuda',requires_grad=True)
                y=model(x)
                assert y.shape==x.shape and y.isfinite().all()
                y.square().mean().backward()
                assert x.grad.isfinite().all()
                for name,p in model.named_parameters():
                    assert p.grad is not None and p.grad.isfinite().all(),(family,width,t,name)
                model.zero_grad(set_to_none=True)
            # Causality test at the mixer boundary, including all projections/conv.
            with torch.no_grad():
                x=torch.randn(2,64,width,device='cuda')
                y=model(x); x[:,32:]=torch.randn_like(x[:,32:])
                torch.testing.assert_close(model(x)[:,:32],y[:,:32],atol=2e-5,rtol=2e-4)
            print(f'PASS mixer forward/backward/causality {family} w{width}',flush=True)
    torch.cuda.synchronize()
    print('PASS ALL LOG-LINEAR GPU CHECKS',flush=True)

if __name__=='__main__': validate()

"""Final-state/gradient checks, including duplicate routes and chunk boundaries."""
import json
import time
import torch
from torch.nn import functional as F
from smat_delta_pool import pool_delta
from zoo_smat_gdn import SmatGDNReset


def reference(k, v, idx, w, beta, cells):
    batch, length, _ = k.shape
    states = k.new_zeros(batch, cells, k.shape[-1], v.shape[-1])
    for t in range(length):
        route = k.new_zeros(batch, cells).scatter_add(1, idx[:, t], w[:, t])
        error = v[:, t, None, :] - torch.einsum('br,bcrp->bcp', k[:, t], states)
        states = states + (route * beta[:, t, None])[:, :, None, None] * k[:, t, None, :, None] * error[:, :, None, :]
    return states


torch.manual_seed(123)
for length, routes, skew in [(19, 4, False), (137, 4, True), (70, 1, False)]:
    shape = (2, length)
    idx = torch.randint(0, 5, (*shape, routes), device='cuda')
    if skew: idx[:, :, :] = 0
    inputs = [F.normalize(torch.randn(*shape, 16, device='cuda'), dim=-1),
              torch.randn(*shape, 16, device='cuda'),
              torch.softmax(torch.randn(*shape, routes, device='cuda'), -1),
              torch.rand(*shape, device='cuda')]
    inputs = [x.requires_grad_() for x in inputs]
    k,v,w,beta = inputs
    actual = pool_delta(k,v,idx,w,beta,7)
    expected = reference(k,v,idx,w,beta,7)
    print("state error", (actual-expected).abs().max().item(), flush=True)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    probe = torch.randn_like(actual)
    grads = torch.autograd.grad((actual * probe).sum(), inputs)
    refs = torch.autograd.grad((expected * probe).sum(), inputs)
    print("gradient errors", [(a-e).abs().max().item() for a,e in zip(grads,refs)], flush=True)
    for a,e in zip(grads,refs): torch.testing.assert_close(a,e,atol=2e-5,rtol=2e-5)
    print('PASS final state and all gradients', length, routes, skew, flush=True)

# Same key overwritten exactly when beta=1; empty cells remain zero.
k = torch.zeros(1,2,16,device='cuda'); k[...,0]=1
v = torch.randn(1,2,16,device='cuda')
f = pool_delta(k,v,torch.zeros(1,2,1,device='cuda',dtype=torch.long),torch.ones(1,2,1,device='cuda'),torch.ones(1,2,device='cuda'),3)
torch.testing.assert_close(f[0,0,0],v[0,1],rtol=1e-5,atol=1e-5)
assert f[:,1:].count_nonzero()==0
print('PASS overwrite and empty cells',flush=True)

results=[]
for mode in ('additive','delta'):
    torch.manual_seed(123)
    m = SmatGDNReset(32,d=3,memory_update=mode).cuda().train()
    count = len(list(m.parameters()))
    for length in (64,128,256):
        for hard in (False,True):
            m._steps = 1100 if hard else 0
            u = torch.randn(32,length,32,device='cuda')
            for it in range(5):
                m.zero_grad(set_to_none=True)
                y=m(u)
                (y.square().mean()+m.get_auxiliary_loss()).backward()
            torch.cuda.synchronize(); start=time.perf_counter()
            for it in range(5):
                m.zero_grad(set_to_none=True)
                y=m(u)
                (y.square().mean()+m.get_auxiliary_loss()).backward()
            torch.cuda.synchronize(); elapsed=(time.perf_counter()-start)/5
            for name,p in m.named_parameters():
                if p.grad is not None: assert torch.isfinite(p.grad).all(),name
            hash_grads=[p.grad for name,p in m.named_parameters() if name.startswith('ca.') and p.grad is not None]
            assert any(g.abs().max()>0 for g in hash_grads)
            assert count==len(list(m.parameters()))
            row=dict(mode=mode,length=length,hard=hard,batch=32,forward_backward_ms=elapsed*1000)
            results.append(row);print(json.dumps(row),flush=True)
    if mode=='delta':
        m.eval();m.disable_g=True
        changed=u.clone();changed[:,:length//2]=torch.randn_like(changed[:,:length//2])
        with torch.no_grad():
            a=m(u);b=m(changed)
            torch.testing.assert_close(a[:,length//2:],b[:,length//2:],atol=0,rtol=0)
            m.disable_g=False;c=m(u)
            assert (c[:,length//2:]-a[:,length//2:]).abs().max()>1e-6
            torch.testing.assert_close(c[:,:length//2],a[:,:length//2],atol=0,rtol=0)
        print('PASS delta mixer isolation, gradients, all lengths',flush=True)
print('ALL DELTA TESTS PASSED',flush=True)
open('results/mqar_gdn_delta_benchmark.json','w').write(json.dumps(results,indent=2))

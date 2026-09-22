"""GPU-only transport algebra, gradient, routing, causality and integration checks."""
import io
import time
import torch
from torch.nn import functional as F
from einops import rearrange
from smat_gdn_transport import boundary_transport
from zoo_gdn_transport import SmatGDNTransport
from content_addr import ContentAssign
from smat_mask import build_mask
from smat_pool_ops import pool_sorted
from smat_write_hash_grad import with_write_hash_gradient


def check_write_hash_gradient():
    """Single-route forward equals dense hard STE, including payload gradients."""
    torch.manual_seed(919)
    b,n,c,r,p=2,11,7,3,4
    k=torch.randn(b,n,r,device='cuda',dtype=torch.float64,requires_grad=True)
    v=torch.randn(b,n,p,device='cuda',dtype=torch.float64,requires_grad=True)
    # Include duplicate destinations (as at clamped coordinates/anchors).
    idx=torch.randint(c,(b,n,4),device='cuda')
    idx[:,0,:]=idx[:,0,:1].clone()
    w=torch.zeros(b,n,4,device='cuda',dtype=torch.float64)
    w[...,0]=1.;w.requires_grad_()
    hard=pool_sorted(k,v,idx[...,:1],w[...,:1].detach(),c)
    actual=with_write_hash_gradient(hard,k,v,idx,w)
    dense=w.new_zeros(b,n,c).scatter_add(-1,idx,w)
    expected=torch.einsum('bnc,bnr,bnp->bcrp',dense,k,v)
    torch.testing.assert_close(actual,hard,atol=0,rtol=0)
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
    probe=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*probe).sum(),(k,v,w))
    ge=torch.autograd.grad((expected*probe).sum(),(k,v,w))
    for a,e in zip(ga,ge): torch.testing.assert_close(a,e,atol=1e-10,rtol=1e-10)
    print('PASS write gradient: single forward route, dense neighbor gradients, payload gradients counted once',flush=True)


def reference(q,k,beta,g,n):
    b,_,h,r=q.shape
    eye=torch.eye(r,device=q.device,dtype=q.dtype).expand(b,h,r,r)
    transforms=g.exp()[...,None,None]*(eye[:,None]-beta[...,None,None]*k[..., :,None]*k[...,None,:])
    product=eye
    keys=[]
    for j in reversed(range(n)):
        keys.append((product@(beta[:,j,:,None]*k[:,j])[...,None]).squeeze(-1))
        product=product@transforms[:,j]
    product=eye
    queries=[]
    for j in range(n,2*n):
        product=transforms[:,j]@product
        queries.append((product.transpose(-1,-2)@q[:,j,...,None]).squeeze(-1))
    return torch.stack(keys[::-1],1),torch.stack(queries,1)


def check_transport():
    for n in (9,32,64,128):
        torch.manual_seed(700+n)
        q,k=[F.normalize(torch.randn(2,2*n,2,16,device='cuda'),dim=-1) for _ in range(2)]
        beta=torch.rand(2,2*n,2,device='cuda')*.8+.1
        g=-torch.rand_like(beta)*.06
        inputs=[x.requires_grad_() for x in (q,k,beta,g)]
        reference_inputs=[x.detach().double().requires_grad_() for x in inputs]
        actual=boundary_transport(*inputs,n)
        expected=reference(*reference_inputs,n)
        for a,e in zip(actual,expected):
            torch.testing.assert_close(a.double(),e,atol=5e-5,rtol=5e-4)
        probes=[torch.randn_like(x) for x in actual]
        ga=torch.autograd.grad(sum((a*p).sum() for a,p in zip(actual,probes)),inputs)
        ge=torch.autograd.grad(sum((e*p.double()).sum() for e,p in zip(expected,probes)),reference_inputs)
        for a,e in zip(ga,ge):
            torch.testing.assert_close(a.double(),e,atol=2e-4,rtol=2e-3)
        print(f'PASS transport n={n}: FP64 ordered products and all Q/K/beta/decay gradients',flush=True)


def check_routed_operator(neighbor_grad=False):
    torch.manual_seed(883)
    b,t,h,r,p,n=2,64,2,16,16,32
    ca=ContentAssign(h,32,build_mask(t,3,chunk=16),codim=1).cuda()
    ca.enable_topk_reads(4,32);ca=ca.cuda()
    ca.sparse_ops,ca.hard_k1,ca.anneal,ca.delta_updates=True,True,1.,False
    ca.write_hash_neighbor_grad=neighbor_grad
    u=torch.randn(b,t,32,device='cuda',requires_grad=True)
    q,k=[F.normalize(torch.randn(b,t,h,r,device='cuda'),dim=-1).requires_grad_() for _ in range(2)]
    v=torch.randn(b,t,h,p,device='cuda',requires_grad=True)
    beta=torch.rand(b,t,h,device='cuda').sigmoid().requires_grad_()
    g=(-torch.rand_like(beta)*.02).requires_grad_()
    keys,queries=boundary_transport(q,k,beta,g,n)
    qf=rearrange(torch.cat((q[:,:n]*0,queries*r**-.5),1),'b t h r -> (b h) t r')
    kf=rearrange(torch.cat((keys,k[:,n:]*0),1),'b t h r -> (b h) t r')
    vf=rearrange(v,'b t h p -> (b h) t p')
    actual=ca(u,qf,kf,vf,n,key_src=u,key_w=rearrange(beta[:,:n],'b t h -> (b h) t'))
    wi,ww=ca._sparse
    ri,rw=ca._topk_reads(u[:,n:])
    assert wi.shape[-1]==1 and ri.shape[-1]==4
    assert (ri.sort(-1).values.diff(dim=-1)>0).all()
    if neighbor_grad:
        ca._cells(u[:,:n],plant_at=True,
                  tok_w=rearrange(beta[:,:n],'b t h -> (b h) t'),retain_neighbors=True)
        wi,ww=ca._sparse
    incidence=ca.M.flatten(0,1)
    mask=incidence[ri[:,:,:,None,None],wi[:,None,None,:,:]]
    routing=((mask*ww[:,None,None,:,:]).sum(-1)*rw[...,None]).sum(2)
    rk,rq=reference(q.double(),k.double(),beta.double(),g.double(),n)
    scores=torch.einsum('bir,bjr->bij',rearrange(rq,'b t h r -> (b h) t r'),
                        rearrange(rk,'b t h r -> (b h) t r'))*r**-.5
    expected=torch.einsum('bij,bjp->bip',routing.double()*scores,vf[:,:n].double())
    torch.testing.assert_close(actual.double(),expected,atol=5e-5,rtol=1e-3)
    probe=torch.randn_like(actual)
    inputs=(u,q,k,v,beta,g,ca.W,ca.gamma,ca.b,ca.W_read,ca.b_read)
    ga=torch.autograd.grad((actual*probe).sum(),inputs,retain_graph=True)
    ge=torch.autograd.grad((expected*probe.double()).sum(),inputs)
    for a,e in zip(ga,ge): torch.testing.assert_close(a,e,atol=3e-4,rtol=3e-3)
    print(f'PASS routed operator: four-read dense reference; neighbor_write_grad={neighbor_grad}',flush=True)


def check_mixer(neighbor_grad=False):
    torch.manual_seed(123)
    kwargs=dict(d=3,headdim=16,n_heads=2,expand_v=1,memory_update='delta',
                memory_key_mode='tied_causal',memory_read_k=4,
                write_hash_neighbor_grad=neighbor_grad)
    model=SmatGDNTransport(32,**kwargs).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    for length in (64,128,256):
        u=torch.randn(4,length,32,device='cuda')
        optimizer.zero_grad(set_to_none=True)
        out=model(u)
        (out.square().mean()+model.get_auxiliary_loss()).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        ca=next(iter(model.ca[str(length)].values()))
        assert ca.last_write_idx.shape[-1]==1
        assert (ca.last_read_idx.sort(-1).values.diff(dim=-1)>0).all()
        for p in (ca.W,ca.W_read,model.kconv.weight,model.gdn.A_log,model.gdn.b_proj.weight):
            assert p.grad is not None and p.grad.abs().max()>0
        optimizer.step()
        print(f'PASS complete mixer length={length}: finite backward, learned routes, transition gradients',flush=True)
    model.eval()
    u=torch.randn(2,64,32,device='cuda')
    # At identical weights the opt-in estimator must preserve the hard output,
    # even when autograd is enabled (where the backward adapter is attached).
    before=model(u).detach()
    model.write_hash_neighbor_grad=not neighbor_grad
    torch.testing.assert_close(model(u).detach(),before,atol=0,rtol=0)
    model.write_hash_neighbor_grad=neighbor_grad
    with torch.no_grad():
        original=model(u)
        for cut in (11,31,32,47):
            changed=u.clone();changed[:,cut+1:]=torch.randn_like(changed[:,cut+1:])
            torch.testing.assert_close(model(changed)[:,:cut+1],original[:,:cut+1],atol=2e-5,rtol=2e-5)
        model.disable_g=True
        isolated=model(u)
        changed=u.clone();changed[:,:32]=torch.randn_like(changed[:,:32])
        torch.testing.assert_close(model(changed)[:,32:],isolated[:,32:],atol=0,rtol=0)
        model.disable_g=False
    data=io.BytesIO();torch.save(model.state_dict(),data);data.seek(0)
    restored=SmatGDNTransport(32,**kwargs).cuda().eval()
    restored.load_state_dict(torch.load(data,weights_only=True))
    with torch.no_grad(): torch.testing.assert_close(restored(u),model(u),atol=0,rtol=0)
    print('PASS causal prefixes, reset isolation and checkpoint restoration',flush=True)
    # Bound practical overhead at the actual training batch/maximum sequence.
    model.train();u=torch.randn(256,256,32,device='cuda')
    times=[]
    for step in range(6):
        torch.cuda.synchronize();start=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss=model(u).square().mean()+model.get_auxiliary_loss();loss.backward();optimizer.step()
        torch.cuda.synchronize()
        if step>1: times.append(time.monotonic()-start)
    print(f'BENCH mixer B256/T256 mean_step_seconds={sum(times)/len(times):.4f}',flush=True)


if __name__=='__main__':
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    check_transport();check_write_hash_gradient()
    for neighbor_grad in (False,True):
        check_routed_operator(neighbor_grad);check_mixer(neighbor_grad)
    print('ALL TRANSPORT GPU CHECKS PASSED',flush=True)

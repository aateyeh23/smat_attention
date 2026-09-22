"""Bounded GPU check of the task-general no-scalar-decay transport option."""
import json
import os
import torch
from torch.nn import functional as F
from fla.modules.l2norm import l2norm
from smat_gdn_transport import boundary_transport
from test_gdn_transport_gpu import reference
from zoo_gdn_transport import SmatGDNTransport

if os.environ.get('GDN_EXPECTED_SOURCES_JSON'):
    from audit_gdn_sources import verify_sources
    verify_sources('gpu-validation')

def hash_parameter(ca):
    return ca.parametrizations.W.original if hasattr(ca, 'parametrizations') else ca.W

torch.set_num_threads(4)
torch.manual_seed(424)
variant = json.loads(os.environ.get('GDN_VARIANT_JSON', '{}'))
if variant.get('baseline', False):
    from zoo_smat_gdn import SmatGDNReset
    width = int(os.environ['ZOO_DM'])
    m = SmatGDNReset(width,d=1,headdim=16,n_heads=2,expand_v=1,
                     memory_update='delta',memory_key_mode='tied_causal').cuda()
    assert m.h == 2 and m.r == 16 and m.p == 16 and len(m.ca) == 0
    for length in (64,128,256):
        m.zero_grad(set_to_none=True)
        x = torch.randn(2,length,width,device='cuda')
        y = m(x)
        direct = m.gdn(x)[0]
        torch.testing.assert_close(y,direct)
        y.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    print(f'PASS native GDN baseline width={width} heads=2 head/state=16; forward/backward and no SMAT',flush=True)
    raise SystemExit(0)
q,k=[F.normalize(torch.randn(2,64,2,16,device='cuda'),dim=-1).requires_grad_()
     for _ in range(2)]
beta=torch.rand(2,64,2,device='cuda').requires_grad_()
g=torch.zeros_like(beta)
actual=boundary_transport(q,k,beta,g,32)
expected=reference(q.double(),k.double(),beta.double(),g.double(),32)
for a,e in zip(actual,expected):
    torch.testing.assert_close(a.double(),e,atol=5e-5,rtol=5e-4)
probes=[torch.randn_like(t) for t in actual]
ga=torch.autograd.grad(sum((a*p).sum() for a,p in zip(actual,probes)),(q,k,beta),retain_graph=True)
ge=torch.autograd.grad(sum((e*p.double()).sum() for e,p in zip(expected,probes)),(q,k,beta))
for a,e in zip(ga,ge): torch.testing.assert_close(a,e,atol=2e-4,rtol=2e-3)
variant = json.loads(os.environ.get('GDN_VARIANT_JSON', '{}'))
options = dict(write_hash_neighbor_grad=True, transport_scalar_decay=False)
options.update(variant.get('kwargs', {}))
width = int(os.environ.get('ZOO_DM', '32'))
m=SmatGDNTransport(width,d=int(os.environ.get('ZOO_DS', '3')),headdim=16,n_heads=2,expand_v=1,memory_update='delta',
    memory_key_mode='tied_causal',memory_read_k=4,**options).cuda()
assert m.h == 2 and m.r == 16 and m.p == 16
for length in (64,128,256):
    m.zero_grad(set_to_none=True)
    x=torch.randn(2,length,width,device='cuda')
    y=m(x);(y.square().mean()+m.get_auxiliary_loss()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    if m.transport_tied_features:
        assert m.memory_qk.weight.grad is not None and m.memory_qk.weight.grad.abs().max()>0
        assert m.kconv.weight.grad is not None and m.kconv.weight.grad.abs().max()>0
    ca=m.ca[str(length)]['1']
    if m.orthogonal_hash:
        torch.testing.assert_close(ca.W @ ca.W.transpose(-1,-2),
            torch.eye(ca.dim,device='cuda').expand(ca.h,-1,-1),atol=2e-6,rtol=2e-5)
    assert ca.last_write_idx.shape[-1]==1 and ca.last_read_idx.shape[-1]==4
    assert (ca.last_read_idx.sort(-1).values.diff(dim=-1)>0).all()
    if not m.memory_plant:
        # Every write, including former anchor positions, must follow its content.
        recorded = ca.last_write_idx.clone()
        with torch.no_grad():
            if m.memory_hash_features == 'gdn_keys':
                segment=x.reshape(4,length//2,width)
                keys,_=m.gdn.k_conv1d(x=m.gdn.k_proj(segment),cache=None,
                    output_final_state=False,cu_seqlens=None)
                keys=l2norm(keys.reshape(2,length,m.h,m.r))
                source=F.linear(keys.flatten(2),m.gdn.k_proj.weight.T)
            else:
                source=m.kconv(F.pad(x.transpose(1,2),(3,0))).transpose(1,2)
            ca._cells(source[:,:length//2],plant_at=False)
            assert torch.equal(recorded,ca._sparse[0])
# Isolating the hash input must preserve outputs and keep the hash/conv trainable.
m.eval()  # Evaluate fixed routing, including candidates with training-only noise.
x=torch.randn(2,64,width,device='cuda',requires_grad=True)
m.zero_grad(set_to_none=True);m.detach_write_hash_input=False
before=m(x).detach();m.get_auxiliary_loss()
m.detach_write_hash_input=True
after=m(x)
torch.testing.assert_close(after.detach(),before,atol=0,rtol=0)
m.get_auxiliary_loss().backward()
assert x.grad is None
if m.memory_hash_features == 'hidden':
    assert m.kconv.weight.grad is not None and m.kconv.weight.grad.abs().max()>0
assert hash_parameter(m.ca['64']['1']).grad is not None and hash_parameter(m.ca['64']['1']).grad.abs().max()>0
print('PASS no-scalar-decay transport reference, gradients and one-write/four-read mixer',flush=True)
print('PASS isolated hash input: identical forward, trainable hash, no auxiliary gradient to input',flush=True)
print('PASS candidate configuration '+json.dumps(options),flush=True)
print(f'PASS model geometry width={width} d={m.d} heads=2 head/state=16',flush=True)
if m.profile_delta:
    from smat_delta_pool import pool_delta
    from smat_write_hash_grad import with_delta_write_hash_gradient
    # Independent time-unrolled reference includes repeated destinations and
    # zero-weight counterfactual routes, plus payload and beta gradients.
    batch,length,cells,routes=2,19,7,4
    idx=torch.randint(cells,(batch,length,routes),device='cuda')
    idx[:,0,:]=idx[:,0,:1].clone()
    keys=F.normalize(torch.randn(batch,length,16,device='cuda'),dim=-1).requires_grad_()
    values=torch.randn(batch,length,16,device='cuda',requires_grad=True)
    beta=torch.rand(batch,length,device='cuda',requires_grad=True)
    weights=torch.zeros(batch,length,routes,device='cuda');weights[...,0]=1.;weights.requires_grad_()
    hard=pool_delta(keys,values,idx[...,:1],weights[...,:1].detach(),beta,cells)
    actual=with_delta_write_hash_gradient(hard,keys,values,idx,weights,beta,cells)
    torch.testing.assert_close(actual,hard,atol=0,rtol=0)
    state=keys.new_zeros(batch,cells,16,16,dtype=torch.float64)
    for t in range(length):
        route=state.new_zeros(batch,cells).scatter_add(1,idx[:,t],weights[:,t].double())
        kt,vt=keys[:,t].double(),values[:,t].double()
        error=vt[:,None,:]-torch.einsum('br,bcrp->bcp',kt,state)
        state=state+(route*beta[:,t,None].double())[:,:,None,None]*kt[:,None,:,None]*error[:,:,None,:]
    torch.testing.assert_close(actual.double(),state,atol=3e-5,rtol=3e-4)
    probe=torch.randn_like(actual)
    inputs=(keys,values,beta,weights)
    ga=torch.autograd.grad((actual*probe).sum(),inputs,retain_graph=True)
    ge=torch.autograd.grad((state*probe.double()).sum(),inputs)
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,atol=4e-5,rtol=4e-4)
    print('PASS delta write adapter: identical one-write forward, FP64 reference and all route/payload/beta gradients',flush=True)
if m.periodic_hash:
    from test_gdn_transport_gpu import check_routed_operator
    check_routed_operator(neighbor_grad=True,periodic_hash=True)
    x=torch.randn(2,64,width,device='cuda')
    before=m(x).detach();m.get_auxiliary_loss()
    for mods in m.ca.values():
        for ca in mods.values():ca.periodic_hash=False
    after=m(x).detach();m.get_auxiliary_loss()
    torch.testing.assert_close(before,after,atol=0,rtol=0)
    print('PASS periodic interpolation preserves the hard forward output exactly',flush=True)
if not m.transport_enabled:
    m.zero_grad(set_to_none=True)
    x=torch.randn(2,64,width,device='cuda')
    m(x).square().mean().backward()  # Task path only, without the balance loss.
    if m.transport_tied_features:
        for p in (m.memory_qk.weight,m.kconv.weight):
            assert p.grad is not None and p.grad.abs().max()>0
    print('PASS untransported memory: retrieval gradients train the tied causal content features',flush=True)
if m.joint_hash_balance:
    from smat_address_reads import write_address_dependence
    # Both distributions have uniform coordinate marginals, but only the
    # diagonal has correlated coordinates. This checks the missing distinction.
    q=5
    uniform_idx=torch.arange(q*q,device='cuda').reshape(1,q*q,1)
    uniform_w=torch.ones_like(uniform_idx,dtype=torch.float64,requires_grad=True)
    diagonal_idx=(torch.arange(q,device='cuda')*(q+1)).reshape(1,q,1)
    diagonal_w=torch.ones_like(diagonal_idx,dtype=torch.float64,requires_grad=True)
    independent=write_address_dependence(uniform_idx,uniform_w,1,q,2)
    correlated=write_address_dependence(diagonal_idx,diagonal_w,1,q,2)
    torch.testing.assert_close(independent,independent.new_zeros(()),atol=1e-12,rtol=0)
    torch.testing.assert_close(correlated,correlated.new_tensor(q).log(),atol=1e-12,rtol=0)
    # Check sparse counts and their gradients against a dense explicit joint.
    idx=torch.randint(q*q,(4,7,4),device='cuda')
    weights=torch.rand(4,7,4,device='cuda',dtype=torch.float64,requires_grad=True)
    actual=write_address_dependence(idx,weights,2,q,2)
    dense=F.one_hot(idx,q*q).to(weights.dtype)*weights[...,None]
    p=dense.sum((1,2)).reshape(2,2,q*q).sum(0)
    p=(p/p.sum(-1,keepdim=True)).reshape(2,q,q)
    expected=(p*(p.clamp_min(1e-9).log()-p.sum(-1,keepdim=True).clamp_min(1e-9).log()
                -p.sum(-2,keepdim=True).clamp_min(1e-9).log())).sum()/2
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
    ga=torch.autograd.grad(actual,weights,retain_graph=True)[0]
    ge=torch.autograd.grad(expected,weights)[0]
    torch.testing.assert_close(ga,ge,atol=1e-12,rtol=1e-10)
    print('PASS joint hash balance: independent vs diagonal occupancy and dense-reference gradients',flush=True)
if m.memory_read_noise:
    ca=m.ca['64']['1']
    assert 0 < ca.read_noise_std < m.memory_read_noise
    x=torch.randn(2,32,width,device='cuda')
    first=ca._topk_reads(x)
    second=ca._topk_reads(x)
    for a,b in zip(first,second): torch.testing.assert_close(a,b,atol=0,rtol=0)
    m.train()
    noisy=ca._topk_reads(x)
    assert (noisy[0].sort(-1).values.diff(dim=-1)>0).all()
    assert not torch.equal(noisy[0],first[0])
    torch.testing.assert_close(noisy[1].sum(-1),torch.ones_like(noisy[1][...,0]))
    m._steps=m.anneal_steps
    m(torch.randn(2,64,width,device='cuda'))
    assert ca.read_noise_std==0.
    m.eval()
    print('PASS read exploration: four distinct weighted reads, deterministic eval, zero noise after annealing',flush=True)
if m.symmetric_write_grad:
    from test_gdn_transport_gpu import check_routed_operator
    check_routed_operator(neighbor_grad=True,symmetric_write_grad=True)
    x=torch.randn(2,64,width,device='cuda')
    before=m(x).detach();m.get_auxiliary_loss()
    for mods in m.ca.values():
        for ca in mods.values():ca.symmetric_write_grad=False
    after=m(x).detach();m.get_auxiliary_loss()
    torch.testing.assert_close(before,after,atol=0,rtol=0)
    print('PASS symmetric write surrogate preserves the hard forward output exactly',flush=True)
if m.orthogonal_hash:
    # The actual AdamW update must preserve orthogonality, including decay.
    optimizer=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.1)
    optimizer.zero_grad(set_to_none=True)
    x=torch.randn(2,64,width,device='cuda')
    y=m(x);(y.square().mean()+m.get_auxiliary_loss()).backward()
    optimizer.step()
    ca=m.ca['64']['1']
    torch.testing.assert_close(ca.W @ ca.W.transpose(-1,-2),
        torch.eye(ca.dim,device='cuda').expand(ca.h,-1,-1),atol=3e-6,rtol=3e-5)
    assert all(torch.isfinite(p).all() for p in m.parameters())
    print('PASS orthogonal hash remains valid after the actual AdamW update',flush=True)
if m.share_hash_across_lengths:
    owner=m.ca['64']['1']
    assert all(m.ca[str(t)]['1'].W is owner.W for t in (64,128,256))
if m.memory_read_address:
    from smat_address_reads import address_read_logits
    ca=m.ca['64']['1']
    x=torch.randn(2,32,width,device='cuda',requires_grad=True)
    m.zero_grad(set_to_none=True)
    ca.read_backend='torch'
    idx,weights=ca._topk_reads(x)
    # Independently enumerate each finite-field line's scalar address mass.
    W=F.normalize(ca.W,dim=-1)
    z=torch.einsum('blm,hkm->bhlk',ca.ln(x),W)*ca.gamma[None,:,None]+ca.b[None,:,None]
    pos=torch.special.ndtr(z)*ca.q
    centers=torch.arange(ca.q,device='cuda')+.5
    probs=(-.5*((pos[...,None]-centers)/ca.address_read_sigma).square()).softmax(-1)
    cell=(probs[...,0,:,None]*probs[...,1,None,:]).flatten(-2)
    expected=torch.stack([cell[...,line.bool()].sum(-1) for line in ca.M.flatten(0,1)],-1)
    if m.memory_concurrent_reads:
        point=pos.clamp(0.,ca.q-1e-4).floor().long()
        point=point[...,0]*ca.q+point[...,1]
        expected=expected*ca.M.flatten(0,1).T[point]
        selected=idx.reshape(2,ca.h,32,4)
        assert ca.M.flatten(0,1).bool()[selected,point[...,None]].all()
        directions=selected//ca.n_cosets
        assert (directions.sort(-1).values.diff(dim=-1)>0).all()
        print('PASS four concurrent reads: distinct directions, all incident on the query hash',flush=True)
    logits=address_read_logits(ca,x)
    torch.testing.assert_close(logits.exp(),expected,atol=2e-6,rtol=2e-5)
    assert idx.shape[-1]==4 and (idx.sort(-1).values.diff(dim=-1)>0).all()
    expected_weights=expected.flatten(0,1).gather(-1,idx)
    expected_weights=expected_weights/expected_weights.sum(-1,keepdim=True)
    torch.testing.assert_close(weights,expected_weights,atol=2e-6,rtol=2e-5)
    (weights*torch.randn_like(weights)).sum().backward()
    assert x.grad is not None and x.grad.abs().max()>0
    assert hash_parameter(ca).grad is not None and hash_parameter(ca).grad.abs().max()>0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    if m.share_hash_across_lengths:
        assert all(m.ca[str(t)]['1'].W is ca.W for t in (64,128,256))
    print('PASS shared-address scores vs enumerated geometry, four reads and query/hash gradients',flush=True)

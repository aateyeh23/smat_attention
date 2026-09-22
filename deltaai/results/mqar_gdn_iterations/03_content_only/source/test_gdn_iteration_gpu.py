"""Bounded GPU check of the task-general no-scalar-decay transport option."""
import json
import os
import torch
from torch.nn import functional as F
from smat_gdn_transport import boundary_transport
from test_gdn_transport_gpu import reference
from zoo_gdn_transport import SmatGDNTransport

torch.set_num_threads(4)
torch.manual_seed(424)
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
m=SmatGDNTransport(32,d=3,headdim=16,n_heads=2,expand_v=1,memory_update='delta',
    memory_key_mode='tied_causal',memory_read_k=4,**options).cuda()
for length in (64,128,256):
    m.zero_grad(set_to_none=True)
    x=torch.randn(2,length,32,device='cuda')
    y=m(x);(y.square().mean()+m.get_auxiliary_loss()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    ca=m.ca[str(length)]['1']
    assert ca.last_write_idx.shape[-1]==1 and ca.last_read_idx.shape[-1]==4
    assert (ca.last_read_idx.sort(-1).values.diff(dim=-1)>0).all()
    if not m.memory_plant:
        # Every write, including former anchor positions, must follow its content.
        recorded = ca.last_write_idx.clone()
        with torch.no_grad():
            source=m.kconv(F.pad(x.transpose(1,2),(3,0))).transpose(1,2)
            ca._cells(source[:,:length//2],plant_at=False)
            assert torch.equal(recorded,ca._sparse[0])
# Isolating the hash input must preserve outputs and keep the hash/conv trainable.
x=torch.randn(2,64,32,device='cuda',requires_grad=True)
m.zero_grad(set_to_none=True);m.detach_write_hash_input=False
before=m(x).detach();m.get_auxiliary_loss()
m.detach_write_hash_input=True
after=m(x)
torch.testing.assert_close(after.detach(),before,atol=0,rtol=0)
m.get_auxiliary_loss().backward()
assert x.grad is None
assert m.kconv.weight.grad is not None and m.kconv.weight.grad.abs().max()>0
assert m.ca['64']['1'].W.grad is not None and m.ca['64']['1'].W.grad.abs().max()>0
print('PASS no-scalar-decay transport reference, gradients and one-write/four-read mixer',flush=True)
print('PASS isolated hash input: identical forward, trainable hash/conv, no auxiliary gradient to input',flush=True)
print('PASS candidate configuration '+json.dumps(options),flush=True)

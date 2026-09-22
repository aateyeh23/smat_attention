"""Bounded GPU check of the task-general no-scalar-decay transport option."""
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
m=SmatGDNTransport(32,d=3,headdim=16,n_heads=2,expand_v=1,memory_update='delta',
    memory_key_mode='tied_causal',memory_read_k=4,write_hash_neighbor_grad=True,
    transport_scalar_decay=False).cuda()
for length in (64,128,256):
    m.zero_grad(set_to_none=True)
    x=torch.randn(2,length,32,device='cuda')
    y=m(x);(y.square().mean()+m.get_auxiliary_loss()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    ca=m.ca[str(length)]['1']
    assert ca.last_write_idx.shape[-1]==1 and ca.last_read_idx.shape[-1]==4
    assert (ca.last_read_idx.sort(-1).values.diff(dim=-1)>0).all()
print('PASS no-scalar-decay transport reference, gradients and one-write/four-read mixer',flush=True)

"""Guard the quoted MQAR head counts before starting a fresh GDN campaign."""
import torch
from zoo_smat_gdn import SmatGDNReset
from mqar_one_gpu import GROUPS

torch.manual_seed(123)
for width, heads in ((16,1),(32,2),(64,2)):
    for d in (1,2,3,4):
        m = SmatGDNReset(width,d=d)
        assert m.h == m.gdn.num_heads == m.gdn.num_v_heads == heads
        assert m.gdn.q_proj.out_features == heads * 16
        assert m.gdn.v_proj.out_features == heads * 16
    print(f'PASS width={width} heads={heads}, d=1,2,3,4',flush=True)
for group in ('gdn','delta','interleaved'):
    assert GROUPS[group][1].name == 'quoted_heads_s123'

u = torch.randn(2,64,32,device='cuda')
m = SmatGDNReset(32,d=1).cuda()
y = m(u)
torch.testing.assert_close(y,m.gdn(u)[0],atol=2e-4,rtol=2e-4)
y.square().mean().backward()
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
print('PASS two-head baseline matches native GDN; finite backward',flush=True)
for mode in ('additive','delta'):
    m = SmatGDNReset(32,d=3,memory_update=mode).cuda()
    y=m(u)
    (y.square().mean()+m.get_auxiliary_loss()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max()>0 for n,p in m.named_parameters() if n.startswith('ca.'))
    print(f'PASS two-head {mode} SMAT forward/backward',flush=True)

m=SmatGDNReset(32,d=3,memory_update='delta',hash_key_shift=1).cuda().eval()
for mods in m.ca.values():
    for ca in mods.values(): ca.anneal=1.
ca=next(iter(m.ca['64'].values()))
# Structural anchor columns have deliberately fixed routes in every SMAT arm.
value_index=next(i for i in range(1,32) if i not in ca.plant_cols.tolist())
u[:,32]=u[:,value_index-1]
y=m(u)
ca=next(iter(m.ca['64'].values()))
assert torch.equal(ca.last_hard_k[:,value_index],ca._sparse[0][:,0,0])
y.square().mean().backward()
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
print('PASS preceding-key routing agrees with identical query key; finite backward',flush=True)
m=SmatGDNReset(32,d=3,memory_update='delta',hash_key_shift=1,
               memory_key_mode='tied_previous').cuda()
y=m(u)
(y.square().mean()+m.get_auxiliary_loss()).backward()
assert m.memory_qk.weight.grad is not None and m.memory_qk.weight.grad.abs().max()>0
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
with torch.no_grad():
    features=m.memory_qk(u)
    torch.testing.assert_close(features[:,32],features[:,value_index-1],atol=0,rtol=0)
print('PASS tied memory query/key features and finite nonzero projection gradients',flush=True)
print('ALL QUOTED-HEAD CHECKS PASSED',flush=True)

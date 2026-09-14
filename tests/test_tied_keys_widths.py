"""Verify the promising tied-memory variant at each requested MQAR width."""
import torch
from smat.mixers.gdn import SmatGDNReset
for width,heads in ((16,1),(32,2),(64,2)):
 torch.manual_seed(123)
 m=SmatGDNReset(width,d=3,memory_update='delta',hash_key_shift=1,memory_key_mode='tied_previous').cuda()
 assert m.h==heads and m.memory_qk.out_features==heads*16
 parameter_count=len(list(m.parameters()))
 optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
 for length in (64,128,256):
  u=torch.randn(2,length,width,device='cuda')
  optimizer.zero_grad(set_to_none=True)
  y=m(u); (y.square().mean()+m.get_auxiliary_loss()).backward()
  assert torch.isfinite(y).all()
  assert m.memory_qk.weight.grad is not None and m.memory_qk.weight.grad.abs().max()>0
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  optimizer.step()
 assert len(list(m.parameters()))==parameter_count
 print(f'PASS tied memory width={width}, heads={heads}, all MQAR lengths, gradients and optimizer',flush=True)

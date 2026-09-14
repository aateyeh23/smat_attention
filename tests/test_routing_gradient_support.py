"""Check whether pruning zero-weight routes changes hard outputs or routing gradients."""
import torch,json
from smat.mixers.gdn import SmatGDNReset
torch.manual_seed(123)
m=SmatGDNReset(32,d=3,memory_update='delta',memory_key_mode='tied_causal').cuda().eval()
x=torch.randn(2,128,32,device='cuda');target=torch.randn_like(x)
rows=[];outs=[];grads=[]
for prune in (True,False):
 m.zero_grad(set_to_none=True)
 for mods in m.ca.values():
  for ca in mods.values():ca.anneal=1.;ca.hard_k1=prune
 y=m(x);loss=(y*target).mean();loss.backward()
 outs.append(y.detach());g={k:p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None};grads.append(g)
 rows.append(dict(pruned=prune,loss=loss.item(),gradient_norms={k:v.norm().item() for k,v in g.items() if k.startswith('ca.128') or k=='kconv.weight'}))
err=(outs[0]-outs[1]).abs().max().item();print('hard_forward_max_error',err,flush=True)
assert torch.allclose(outs[0],outs[1],atol=2e-5,rtol=2e-5)
changes={k:(v-grads[1][k]).norm().item() for k,v in grads[0].items() if k.startswith('ca.128') or k=='kconv.weight'}
assert any(v>1e-7 for v in changes.values())
print(json.dumps(dict(rows=rows,gradient_difference_norms=changes,hard_forward_max_error=err),indent=2),flush=True)

import torch
from torch.nn import functional as F
from smat_delta_pool import pool_delta

torch.manual_seed(123)
idx=torch.randint(0,7,(4,137,4),device='cuda')
inputs=[F.normalize(torch.randn(4,137,16,device='cuda'),dim=-1),
        torch.randn(4,137,32,device='cuda'),
        torch.softmax(torch.randn(4,137,4,device='cuda'),-1),
        torch.rand(4,137,device='cuda')]
k,v,w,beta=[x.requires_grad_() for x in inputs]
a=pool_delta(k,v,idx,w,beta,7)
b=pool_delta(k,v,idx,w,beta,7,max_chunks=42)
torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
probe=torch.randn_like(a)
ga=torch.autograd.grad((a*probe).sum(),inputs)
gb=torch.autograd.grad((b*probe).sum(),inputs)
for x,y in zip(ga,gb): torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-5)
print('PASS sharded versus unsharded state and all gradients',flush=True)

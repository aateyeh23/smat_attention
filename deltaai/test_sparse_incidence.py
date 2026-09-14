"""Compare sparse incidence forward/backward with the existing dense operation."""
import json,time
import numpy as np
import torch
from content_addr import _grouped_planes
from smat_incidence_sparse import aggregate_profiles
rows=[]
for q,dim in ((3,1),(5,2),(7,3),(13,2)):
 _,grouped=_grouped_planes(q,dim);D,O,J=grouped.shape;N=q**dim
 M=np.zeros((D,O,N),np.float32)
 for e in range(D):
  for o in range(O):M[e,o,grouped[e,o]]=1
 cosets=torch.tensor(M.argmax(1),device='cuda').contiguous()
 members=torch.tensor(grouped,device='cuda').contiguous()
 for dtype in (torch.float32,torch.bfloat16):
  f=torch.randn(2,N,16,32,device='cuda',dtype=dtype,requires_grad=True)
  matrix=torch.tensor(M,device='cuda',dtype=dtype)
  target=torch.randn(2,D*O,16,32,device='cuda',dtype=dtype)
  sparse=aggregate_profiles(f,members,cosets)
  dense=torch.einsum('eox,bxrp->beorp',matrix,f).reshape_as(sparse)
  gs=torch.autograd.grad((sparse*target).sum(),f)[0]
  gd=torch.autograd.grad((dense*target).sum(),f)[0]
  atol,rtol=(2e-5,2e-5) if dtype==torch.float32 else (.25,.02)
  assert torch.allclose(sparse,dense,atol=atol,rtol=rtol),(q,dim,dtype,'forward')
  assert torch.allclose(gs,gd,atol=atol,rtol=rtol),(q,dim,dtype,'backward')
  row=dict(q=q,dim=dim,dtype=str(dtype),forward_max_error=(sparse-dense).abs().max().item(),backward_max_error=(gs-gd).abs().max().item(),edges=D*O*J,dense_entries=D*O*N)
  rows.append(row);print(json.dumps(row),flush=True)
print('PASS sparse incidence forward/backward',flush=True)

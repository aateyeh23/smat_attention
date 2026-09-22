"""GPU numerical equivalence and timing for the fused write-hash contraction."""
import json
import torch
from smat_write_hash_triton import write_hash_gradient


def check():
    torch.manual_seed(9)
    for dim,routes in ((16,2),(64,2),(128,2),(16,4),(64,4),(64,8),(128,8)):
        b,l,cells=2,67,31
        # Strided time slices match actual pooling inputs at the half boundary.
        k=torch.randn(b,l*2,dim,device='cuda')[:,:l]
        v=torch.randn_like(k)
        ids=torch.randint(cells,(b,l,routes),device='cuda')
        dm=torch.randn(b,cells,dim,dim,device='cuda')
        out=write_hash_gradient(dm,k,v,ids,torch.float32)
        selected=dm[torch.arange(b,device='cuda')[:,None,None],ids]
        ref=torch.einsum('blrpq,blp,blq->blr',selected,k,v)
        torch.testing.assert_close(out,ref,atol=2e-4,rtol=1e-4)
        print('PASS WRITE GRADIENT '+json.dumps(dict(dim=dim,routes=routes,max_abs=(out-ref).abs().max().item())),flush=True)
    print('PASS ALL FUSED WRITE-GRADIENT CHECKS',flush=True)

if __name__=='__main__':check()

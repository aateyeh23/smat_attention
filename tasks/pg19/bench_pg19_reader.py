"""Numerical and throughput checks for the tiled GDN reader."""
import os
import torch
from smat_read_triton import read_topk as reference
from smat_read_tiled import read_topk as candidate
from bench_pg19_transport import emit, measure

torch.set_num_threads(4)
torch.manual_seed(817)
for b,m,n,r,p in ((2,63,17,16,32),(2,127,29,128,128),(12,8192,97,128,128),(12,8192,552,128,128)):
    q=torch.randn(b,m,r,device='cuda',requires_grad=True)
    f=torch.randn(b,n,r,p,device='cuda',requires_grad=True)
    w=torch.rand(b,m,4,device='cuda',requires_grad=True)
    idx=torch.randint(n,(b,m,4),device='cuda')
    probe=torch.randn(b,m,p,device='cuda')
    def compute(fn):
        y=fn(q,idx,w,f)
        grads=torch.autograd.grad(y,(q,w,f),probe)
        return (y,*grads)
    expected=compute(reference)
    baseline=measure(lambda:compute(reference))
    emit('reader_reference',shape=[b,m,n,r,p],ms=baseline)
    if m<200:
        gathered=f[torch.arange(b,device='cuda')[:,None,None],idx]
        y=torch.einsum('bmr,bmkrp,bmk->bmp',q,gathered,w)
        grads=torch.autograd.grad(y,(q,w,f),probe)
        for a,e in zip(expected,(y,*grads)):
            torch.testing.assert_close(a,e,atol=2e-4,rtol=2e-4)
    for bm in (16,32,64):
        os.environ['SMAT_READ_TILE_ROWS']=str(bm)
        result=compute(candidate)
        differences=[]
        for a,e in zip(result,expected):
            torch.testing.assert_close(a,e,atol=3e-4,rtol=3e-4)
            differences.append(dict(max_abs=(a-e).abs().max().item(),relative_l2=((a-e).norm()/e.norm()).item()))
        ms=measure(lambda:compute(candidate))
        emit('reader_candidate',shape=[b,m,n,r,p],bm=bm,ms=ms,speedup=baseline/ms,differences=differences)
    del q,f,w,idx,probe,expected,result
    torch.cuda.empty_cache()
emit('reader_complete')

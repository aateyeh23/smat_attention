import torch
from smat_write_hash_triton import write_hash_gradient as reference
from smat_write_hash_tiled import write_hash_gradient as candidate
from bench_pg19_transport import emit, measure

torch.set_num_threads(4)
torch.manual_seed(173)
for b,l,n,r,routes in ((2,67,31,16,2),(2,67,31,64,4),(2,67,31,128,8),
                       (12,8192,97,128,2),(12,8192,529,128,4)):
    k=torch.randn(b,l*2,r,device='cuda')[:,:l]
    v=torch.randn_like(k)
    g=torch.randn(b,n,r,r,device='cuda')
    idx=torch.randint(n,(b,l,routes),device='cuda')
    args=(g,k,v,idx,torch.float32)
    expected=reference(*args);actual=candidate(*args)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-4)
    base=measure(lambda:reference(*args));ms=measure(lambda:candidate(*args))
    emit('write_benchmark',shape=[b,l,n,r,routes],reference_ms=base,optimized_ms=ms,speedup=base/ms,
         max_abs=(actual-expected).abs().max().item(),relative_l2=((actual-expected).norm()/expected.norm()).item())
emit('write_complete')

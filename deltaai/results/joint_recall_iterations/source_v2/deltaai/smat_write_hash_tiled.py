"""Group neighbor hash gradients by memory cell to reuse state-gradient tiles."""
import torch
import triton
import triton.language as tl
from smat_read_triton import _layout, _find_cell


@triton.jit
def _contract_group(G,Q,V,Order,Starts,Substarts,Out,NC:tl.constexpr,K:tl.constexpr,
                    R:tl.constexpr,P:tl.constexpr,BM:tl.constexpr,BS:tl.constexpr):
    group=tl.program_id(0);ip=tl.program_id(1)
    if group<tl.load(Substarts+NC):
        cell=_find_cell(Substarts,group,NC)
        start=tl.load(Starts+cell);stop=tl.load(Starts+cell+1)
        rows=start+(group-tl.load(Substarts+cell))*BM+tl.arange(0,BM)
        ev=tl.load(Order+rows,rows<stop,other=0)
        p=ip*BS+tl.arange(0,BS)
        acc=tl.zeros((BM,BS),tl.float32)
        for off in range(tl.cdiv(R,BS)):
            r=off*BS+tl.arange(0,BS)
            q=tl.load(Q+(ev//K)[:,None]*R+r[None,:],(rows<stop)[:,None]&(r<R)[None,:],other=0)
            g=tl.load(G+cell*R*P+r[:,None]*P+p[None,:],(r<R)[:,None]&(p<P)[None,:],other=0)
            acc+=tl.dot(q,g,input_precision='tf32x3')
        v=tl.load(V+(ev//K)[:,None]*P+p[None,:],(rows<stop)[:,None]&(p<P)[None,:],other=0)
        tl.store(Out+ev*tl.cdiv(P,BS)+ip,tl.sum(acc*v,axis=1),rows<stop)


def write_hash_gradient(memory_grad,keys,values,indices,dtype):
    b,length,routes=indices.shape
    cells,r,p=memory_grad.shape[1:]
    keys,values,indices,memory_grad=[t.contiguous() for t in (keys,values,indices,memory_grad)]
    bm=32;bs=64
    order,starts,substarts,groups=_layout(indices,cells,bm)
    parts=torch.empty(b,length,routes,triton.cdiv(p,bs),device=keys.device,dtype=torch.float32)
    _contract_group[(groups,triton.cdiv(p,bs))](memory_grad,keys,values,order,starts,substarts,parts,
                                            b*cells,routes,r,p,bm,bs,num_warps=4)
    return parts.sum(-1).to(dtype)

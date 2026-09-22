"""FP32 top-k reader with 64x64 state tiles for 128-dimensional GDN heads."""
import os
import torch
import triton
import triton.language as tl
from smat_read_triton import _layout, _find_cell, _reduce_pairs


@triton.jit
def _forward(Q,W,F,Order,Starts,Substarts,Y,NC:tl.constexpr,K:tl.constexpr,
             R:tl.constexpr,P:tl.constexpr,BM:tl.constexpr,BS:tl.constexpr):
    group=tl.program_id(0); ip=tl.program_id(1)
    if group<tl.load(Substarts+NC):
        cell=_find_cell(Substarts,group,NC)
        start=tl.load(Starts+cell);stop=tl.load(Starts+cell+1)
        rows=start+(group-tl.load(Substarts+cell))*BM+tl.arange(0,BM)
        ev=tl.load(Order+rows,rows<stop,other=0)
        w=tl.load(W+ev,rows<stop,other=0)
        p=ip*BS+tl.arange(0,BS)
        acc=tl.zeros((BM,BS),tl.float32)
        for off in range(tl.cdiv(R,BS)):
            r=off*BS+tl.arange(0,BS)
            q=tl.load(Q+(ev//K)[:,None]*R+r[None,:],(rows<stop)[:,None]&(r<R)[None,:],other=0)
            f=tl.load(F+cell*R*P+r[:,None]*P+p[None,:],(r<R)[:,None]&(p<P)[None,:],other=0)
            acc+=tl.dot(q*w[:,None],f,input_precision='tf32x3')
        tl.store(Y+ev[:,None]*P+p[None,:],acc,(rows<stop)[:,None]&(p<P)[None,:])


@triton.jit
def _backward_query(Q,W,F,DY,Order,Starts,Substarts,DQ,DW,NC:tl.constexpr,K:tl.constexpr,
                    R:tl.constexpr,P:tl.constexpr,BM:tl.constexpr,BS:tl.constexpr):
    group=tl.program_id(0);ir=tl.program_id(1)
    if group<tl.load(Substarts+NC):
        cell=_find_cell(Substarts,group,NC)
        start=tl.load(Starts+cell);stop=tl.load(Starts+cell+1)
        rows=start+(group-tl.load(Substarts+cell))*BM+tl.arange(0,BM)
        ev=tl.load(Order+rows,rows<stop,other=0)
        r=ir*BS+tl.arange(0,BS)
        acc=tl.zeros((BM,BS),tl.float32)
        for off in range(tl.cdiv(P,BS)):
            p=off*BS+tl.arange(0,BS)
            dy=tl.load(DY+(ev//K)[:,None]*P+p[None,:],(rows<stop)[:,None]&(p<P)[None,:],other=0)
            f=tl.load(F+cell*R*P+r[:,None]*P+p[None,:],(r<R)[:,None]&(p<P)[None,:],other=0)
            acc+=tl.dot(dy,tl.trans(f),input_precision='tf32x3')
        w=tl.load(W+ev,rows<stop,other=0)
        q=tl.load(Q+(ev//K)[:,None]*R+r[None,:],(rows<stop)[:,None]&(r<R)[None,:],other=0)
        tl.store(DQ+ev[:,None]*R+r[None,:],acc*w[:,None],(rows<stop)[:,None]&(r<R)[None,:])
        tl.store(DW+ev*tl.cdiv(R,BS)+ir,tl.sum(acc*q,axis=1),rows<stop)


@triton.jit
def _backward_state(Q,W,DY,Order,Starts,DF,K:tl.constexpr,R:tl.constexpr,P:tl.constexpr,
                    BM:tl.constexpr,BS:tl.constexpr):
    cell=tl.program_id(0); ir=tl.program_id(1); ip=tl.program_id(2)
    start=tl.load(Starts+cell);stop=tl.load(Starts+cell+1)
    r=ir*BS+tl.arange(0,BS);p=ip*BS+tl.arange(0,BS)
    acc=tl.zeros((BS,BS),tl.float32)
    for off in range(start,stop,BM):
        rows=off+tl.arange(0,BM)
        ev=tl.load(Order+rows,rows<stop,other=0)
        q=tl.load(Q+(ev//K)[:,None]*R+r[None,:],(rows<stop)[:,None]&(r<R)[None,:],other=0)
        dy=tl.load(DY+(ev//K)[:,None]*P+p[None,:],(rows<stop)[:,None]&(p<P)[None,:],other=0)
        w=tl.load(W+ev,rows<stop,other=0)
        acc+=tl.dot(tl.trans(q*w[:,None]),dy,input_precision='tf32x3')
    tl.store(DF+cell*R*P+r[:,None]*P+p[None,:],acc,(r<R)[:,None]&(p<P)[None,:])


class _TiledRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,idx,w,f,bm):
        b,m,r=q.shape;n,p,k=f.shape[1],f.shape[-1],idx.shape[-1]
        bs=64
        order,starts,substarts,groups=_layout(idx,n,bm)
        pairs=q.new_empty(b,m,k,p);y=q.new_empty(b,m,p)
        _forward[(groups,triton.cdiv(p,bs))](q,w,f,order,starts,substarts,pairs,b*n,k,r,p,bm,bs,num_warps=4)
        _reduce_pairs[(b*m,)](pairs,y,k,p,triton.next_power_of_2(k),triton.next_power_of_2(p),num_warps=4)
        ctx.save_for_backward(q,w,f,order,starts,substarts)
        ctx.shape=(b,m,n,k,r,p,bm,bs,groups)
        return y

    @staticmethod
    def backward(ctx,dy):
        q,w,f,order,starts,substarts=ctx.saved_tensors
        b,m,n,k,r,p,bm,bs,groups=ctx.shape
        dy=dy.contiguous();pairs=q.new_empty(b,m,k,r)
        dq=torch.empty_like(q);df=torch.empty_like(f)
        parts=q.new_empty(b,m,k,triton.cdiv(r,bs))
        _backward_query[(groups,triton.cdiv(r,bs))](q,w,f,dy,order,starts,substarts,pairs,parts,
            b*n,k,r,p,bm,bs,num_warps=4)
        _reduce_pairs[(b*m,)](pairs,dq,k,r,triton.next_power_of_2(k),triton.next_power_of_2(r),num_warps=4)
        _backward_state[(b*n,triton.cdiv(r,bs),triton.cdiv(p,bs))](q,w,dy,order,starts,df,k,r,p,bm,bs,num_warps=4)
        return dq,None,parts.sum(-1),df,None


def read_topk(q,idx,weights,memory):
    if not q.is_cuda or any(t.dtype!=torch.float32 for t in (q,weights,memory)):
        raise ValueError('Tiled reader requires CUDA FP32 tensors')
    bm=int(os.environ.get('SMAT_READ_TILE_ROWS','64'))
    if bm not in (16,32,64):raise ValueError('Unsupported reader tile rows')
    return _TiledRead.apply(q.contiguous(),idx.contiguous(),weights.contiguous(),memory.contiguous(),bm)

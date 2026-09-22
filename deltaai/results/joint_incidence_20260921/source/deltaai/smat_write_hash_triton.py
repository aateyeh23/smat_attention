"""Fused contraction of neighbor cell gradients with each token's outer product."""
import torch
import triton
import triton.language as tl


@triton.jit
def _contract(G,K,V,I,O,L:tl.constexpr,R:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,
              G0:tl.constexpr,G1:tl.constexpr,G2:tl.constexpr,G3:tl.constexpr,
              K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
              V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,
              I0:tl.constexpr,I1:tl.constexpr,I2:tl.constexpr,
              BK:tl.constexpr,BV:tl.constexpr):
    row=tl.program_id(0)
    b=row//(L*R);t=(row//R)%L;r=row%R
    cell=tl.load(I+b*I0+t*I1+r*I2)
    k=tl.arange(0,BK);v=tl.arange(0,BV)
    key=tl.load(K+b*K0+t*K1+k*K2,k<DK,other=0).to(tl.float32)
    val=tl.load(V+b*V0+t*V1+v*V2,v<DV,other=0).to(tl.float32)
    grad=tl.load(G+b*G0+cell*G1+k[:,None]*G2+v[None,:]*G3,
                 (k[:,None]<DK)&(v[None,:]<DV),other=0).to(tl.float32)
    result=tl.sum(tl.sum(grad*key[:,None],axis=0)*val,axis=0)
    tl.store(O+row,result)


def write_hash_gradient(memory_grad,keys,values,indices,dtype):
    b,length,routes=indices.shape
    _,cells,dk,dv=memory_grad.shape
    output=torch.empty(indices.shape,device=keys.device,dtype=dtype)
    _contract[(b*length*routes,)](memory_grad,keys,values,indices,output,
        length,routes,dk,dv,*memory_grad.stride(),*keys.stride(),*values.stride(),*indices.stride(),
        triton.next_power_of_2(dk),triton.next_power_of_2(dv),num_warps=8 if dk*dv>=16384 else 4,
        enable_fp_fusion=False)
    return output

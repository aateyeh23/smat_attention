"""Sparse profile-to-type aggregation with an atomics-free transpose backward.

Each profile belongs to exactly one coset for each direction. Work is proportional
 to incidence edges times state size, with no dense incidence matmul or edge-state
 intermediate. Inputs are contiguous GPU tensors; float32 accumulation.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _forward(F, Members, U, N:tl.constexpr, H:tl.constexpr, J:tl.constexpr,
             Q:tl.constexpr, BLOCK:tl.constexpr):
 z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 h=tl.program_id(1);b=tl.program_id(2)
 acc=tl.full((BLOCK,),0,tl.float32)
 for j in range(J):
  x=tl.load(Members+h*J+j)
  acc+=tl.load(F+(b*N+x)*Q+z,z<Q,0).to(tl.float32)
 tl.store(U+(b*H+h)*Q+z,acc,z<Q)

@tr.jit
def _backward(DU, Cosets, DF, N:tl.constexpr, D:tl.constexpr, O:tl.constexpr,
              Q:tl.constexpr, BLOCK:tl.constexpr):
 z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 x=tl.program_id(1);b=tl.program_id(2)
 acc=tl.full((BLOCK,),0,tl.float32)
 for e in range(D):
  o=tl.load(Cosets+e*N+x)
  acc+=tl.load(DU+(b*D*O+e*O+o)*Q+z,z<Q,0).to(tl.float32)
 tl.store(DF+(b*N+x)*Q+z,acc,z<Q)

class _Incidence(torch.autograd.Function):
 @staticmethod
 def forward(ctx,F,members,cosets):
  B,N,R,P=F.shape;D,O,J=members.shape
  assert F.is_cuda and F.is_contiguous() and members.is_contiguous() and cosets.is_contiguous()
  assert cosets.shape==(D,N)
  U=torch.empty((B,D*O,R,P),device=F.device,dtype=F.dtype)
  _forward[(tr.cdiv(R*P,128),D*O,B)](F,members,U,N,D*O,J,R*P,128)
  ctx.save_for_backward(cosets);ctx.shape=(B,N,R,P);ctx.D=D;ctx.O=O
  return U
 @staticmethod
 def backward(ctx,du):
  (cosets,)=ctx.saved_tensors;B,N,R,P=ctx.shape
  df=torch.empty(ctx.shape,device=du.device,dtype=du.dtype)
  _backward[(tr.cdiv(R*P,128),N,B)](du.contiguous(),cosets,df,N,ctx.D,ctx.O,R*P,128)
  return df,None,None

def aggregate_profiles(F,members,cosets):
 return _Incidence.apply(F,members,cosets)

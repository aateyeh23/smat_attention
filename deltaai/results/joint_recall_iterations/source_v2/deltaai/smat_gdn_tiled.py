"""Shape-specific FLA backend: use smaller FP32 backward tiles on H100.

Launcher adapted from FLA's ops/common/chunk_o.py (MIT), copyright
2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li. The original Triton kernel
and all recurrence/gradient equations are retained.
"""
import os
import torch
import triton
from fla.ops.backends import BaseBackend, BackendRegistry
from fla.ops.common.chunk_o import chunk_bwd_kernel_dqkwg, TRITON_ABOVE_3_7_1


class _TransportBackend(BaseBackend):
    backend_type='smat_gdn_fp32_tiled'
    env_var='SMAT_GDN_TILED_BACKWARD'
    default_enable=False
    priority=-10

    def chunk_bwd_dqkwg_verifier(self,q,k,v,**kwargs):
        valid=(TRITON_ABOVE_3_7_1 and q.is_cuda and q.dtype==torch.float32 and k.dtype==torch.float32
               and v.dtype==torch.float32 and q.shape[-1]==128 and v.shape[-1]==128
               and q.shape[2]==k.shape[2]==v.shape[2]
               and kwargs.get('cu_seqlens') is None
               and torch.cuda.get_device_capability(q.device)==(9,0))
        return valid,'Only equal-length FP32 128-dimensional H100 transport is tuned'

    def chunk_bwd_dqkwg(self,q,k,v,do,h,dh,w=None,g=None,g_gamma=None,dv=None,
                       scale=None,state_v_first=False,cu_seqlens=None,chunk_size=64,
                       chunk_indices=None):
        b,t,heads,r=k.shape;hv,p=v.shape[2:]
        bk=bv=64;nk=triton.cdiv(r,bk);nt=triton.cdiv(t,chunk_size)
        dq=torch.empty_like(q);dk=torch.empty_like(k)
        dw=torch.empty_like(w) if w is not None else None
        dg=torch.empty(nk,*g.shape,device=g.device,dtype=torch.float32) if g is not None else None
        chunk_bwd_kernel_dqkwg[(nk,nt,b*hv)](
            q=q,k=k,v=v,g=g,g_gamma=g_gamma,h=h,do=do,dh=dh,dw=dw,dq=dq,dk=dk,dv=dv,dg=dg,
            cu_seqlens=None,chunk_indices=chunk_indices,scale=scale,B=b,T=t,H=heads,HV=hv,
            K=r,V=p,BT=chunk_size,BK=bk,BV=bv,STATE_V_FIRST=state_v_first)
        if dg is not None:dg=dg.sum(0)
        return dq,dk,dw,dg


def enable():
    BackendRegistry.ensure_initialized('common')
    registry=BackendRegistry._registries.get('common')
    if registry is None:registry=BackendRegistry('common')
    if _TransportBackend.backend_type not in registry._backends:
        registry.register(_TransportBackend())
    os.environ['SMAT_GDN_TILED_BACKWARD']='1'


def disable():
    os.environ['SMAT_GDN_TILED_BACKWARD']='0'

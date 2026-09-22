"""Exact dense Log-Linear operator, checkpointed by batch to bound memory.

Same operator as the completed Table 2 SCF campaign; 13 levels cover joint
recall's maximum length3076. Checkpointing changes storage, not the objective
or effective optimizer batch. No sequence truncation or detached gradients.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from zoo_log_linear import log_linear as dense_log_linear


def dense_core(q,k,v,g,lam,beta=None):
    dtype=v.dtype
    with torch.autocast('cuda',enabled=False):
        q,k,v,g,lam=[x.float() for x in (q,k,v,g,lam)]
        if beta is not None:
            beta=beta.float()
            q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)
            k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
        return dense_log_linear(q,k,v,g,lam,beta).to(dtype)


def log_linear(q,k,v,g,lam,beta=None):
    # Bound each dense matrix to 256M entries; the outer model retains the
    # full optimizer batch and its original dropout draws.
    chunk=min(q.shape[0],max(1,256_000_000//(q.shape[2]*q.shape[1]**2)))
    outputs=[]
    for start in range(0,q.shape[0],chunk):
        arguments=[x[start:start+chunk] for x in (q,k,v,g,lam)]
        if beta is not None:arguments.append(beta[start:start+chunk])
        if torch.is_grad_enabled() and q.shape[1]>256:
            outputs.append(checkpoint(dense_core,*arguments,use_reentrant=False,preserve_rng_state=False))
        else:
            outputs.append(dense_core(*arguments))
    return torch.cat(outputs,dim=0)


class JointLogLinearMixer(nn.Module):
    def __init__(self, d_model, layer_idx=0, family='gdn', **kwargs):
        super().__init__()
        self.family = family
        if family == 'gdn':
            from fla.layers.gated_deltanet import GatedDeltaNet
            self.backbone = GatedDeltaNet(hidden_size=d_model, head_dim=16,
                num_heads=1 if d_model == 16 else 2, expand_v=1, mode='chunk',
                use_gate=True, use_short_conv=True, layer_idx=layer_idx)
            self.heads = self.backbone.num_heads
        elif family == 'mamba2':
            from zoology.mixers.mamba2 import Mamba2
            self.backbone = Mamba2(d_model=d_model, d_state=16, d_conv=4,
                expand=2, headdim=16, ngroups=1, chunk_size=64, use_mem_eff_path=False)
            self.heads = self.backbone.nheads
        else:
            raise ValueError(family)
        self.l_proj = nn.Linear(d_model, self.heads * 13, bias=False)
        self.L = nn.Parameter(torch.ones(self.heads, 13))

    def forward(self, u, **kwargs):
        b, t, _ = u.shape
        if t > 4096:
            raise ValueError('Joint Log-Linear supports T <= 4096')
        lam = F.softplus(self.l_proj(u).reshape(b, t, self.heads, 13) * self.L)
        m = self.backbone
        if self.family == 'gdn':
            streams = []
            for proj, conv in ((m.q_proj,m.q_conv1d), (m.k_proj,m.k_conv1d), (m.v_proj,m.v_conv1d)):
                x, _ = conv(x=proj(u), cache=None, output_final_state=False, cu_seqlens=None)
                streams.append(x.reshape(b,t,self.heads,16))
            q, k, v = streams
            beta = m.b_proj(u).sigmoid()
            if m.allow_neg_eigval:
                beta = beta * 2
            g = -m.A_log.float().exp() * F.softplus(m.a_proj(u).float() + m.dt_bias)
            # Upstream SCALE_LAMBDA=True divides lambda by K**-.5, cancelling q scale.
            y = log_linear(q, k, v, g, lam, beta)
            y = m.o_norm(y, m.g_proj(u).reshape(b,t,self.heads,16))
            return m.o_proj(y.reshape(b,t,-1))
        z, xbc, dt = torch.split(m.in_proj(u),
            [m.d_ssm, m.d_ssm + 2*m.ngroups*m.d_state, m.nheads], dim=-1)
        xbc = m.act(m.conv1d(xbc.transpose(1,2)).transpose(1,2))[:, :t]
        x, k, q = torch.split(xbc, [m.d_ssm,m.ngroups*m.d_state,m.ngroups*m.d_state], dim=-1)
        x = x.reshape(b,t,self.heads,16)
        k, q = (a.reshape(b,t,1,16).expand(b,t,self.heads,16) for a in (k,q))
        dt = F.softplus(dt + m.dt_bias)
        g = -m.A_log.float().exp() * dt
        y = log_linear(q,k,x * dt[...,None],g,lam)
        y = y + x * m.D[None,None,:,None]
        return m.out_proj(m.norm(y.reshape(b,t,-1),z))

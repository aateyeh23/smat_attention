"""Joint recall Log-Linear: original projections, upstream WEAK binary kernels."""
import sys
import types
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

# Load the operator package without the unrelated Transformers registrations.
_UPSTREAM = Path(__file__).resolve().parent.parent/'results/paper_seeds_20260920/source/loglinear_upstream/hattention'
for name in ['paper_legacy_fla', 'paper_legacy_fla.ops', 'paper_legacy_fla.modules', 'paper_legacy_fla.ops.common']:
    if name not in sys.modules:
        package=types.ModuleType(name)
        package.__path__=[str(_UPSTREAM.parent.joinpath(*name.split('.')))]
        sys.modules[name]=package
if 'hattention' not in sys.modules:
    package=types.ModuleType('hattention'); package.__path__=[str(_UPSTREAM)]
    sys.modules['hattention']=package
from hattention.base import HType,HStruct
import hattention.base as _base


@lru_cache(maxsize=16)
def _levels_matrix(length, base, htype, dtype, device, default=-1, clamp_min=None, **kwargs):
    # Same WEAK dyadic lookup, generated locally instead of an upstream
    # developer's hardcoded /export/share lookup-table file.
    assert base == 2 and htype == HType.WEAK
    from zoo_log_linear import level_indices
    levels=level_indices(length,device).to(dtype)
    levels=levels.masked_fill(torch.ones_like(levels,dtype=torch.bool).triu(1),default)
    return levels if clamp_min is None else levels.clamp_min(clamp_min)


_base.make_levels_matrix = _levels_matrix
from hattention.kernel import hattention_kernel


def log_linear(q,k,v,g,lam,beta=None):
    # Upstream kernels require multiples of 64. Zero padding adds no state
    # coordinates; causal future padding cannot influence retained outputs.
    length, value_dim = v.shape[1], v.shape[-1]
    padding = (-length) % 64
    lam=lam[...,:((length+padding-1).bit_length()+1)]
    q,k,v = [F.pad(x, (0, 64-x.shape[-1], 0, 0, 0, padding)) for x in (q,k,v)]
    g=F.pad(g,(0,0,0,padding));lam=F.pad(lam,(0,0,0,0,0,padding))
    if beta is not None:
        beta=F.pad(beta,(0,0,0,padding))
    output = hattention_kernel(q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
        b=None if beta is None else beta.contiguous(), g=g.contiguous().float(),
        l=lam.to(q.dtype).contiguous(), scale=None if beta is None else 1.,
        head_first=False, level_base=2, htype=HType.WEAK,
        hstruct=HStruct.MAMBA2 if beta is None else HStruct.GDELTA,
        use_qk_l2norm_in_kernel=beta is not None)
    return output[:,:length,:,:value_dim].contiguous()


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

"""Base-2 WEAK Log-Linear operators for short MQAR, with dense GPU execution.

Equations: HanGuo97/log-linear-attention@7f8644159c1406fae1ad863829a5b3a4fbf63022,
hattention/base.py hattention_materialized_v2 / hattention_materialized_dplr_v2.
This is NOT the upstream chunkwise Triton implementation. Backbone projections,
convolutions, norms and initialization follow our existing Zoology controls.
"""
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_COMMIT = '7f8644159c1406fae1ad863829a5b3a4fbf63022'
NUM_LEVELS = 9  # ceil(log2(max MQAR length 256)) + 1; shared across lengths


def level_indices(length, device):
    t = torch.arange(length, device=device)
    xor = t[:, None] ^ t[None, :]
    # bit_length(t XOR s): the upstream WEAK binary recursion, diagonal zero.
    levels = torch.zeros_like(xor)
    for bit in range((length - 1).bit_length()):
        levels = torch.where(xor >= 2 ** bit, bit + 1, levels)
    return levels


def log_linear(q, k, v, g, lam, beta=None):
    """Inputs B,T,H,D; gates B,T,H; positive lambda B,T,H,L.

    For delta transitions, solve the unit-lower system from upstream's dense
    DPLR reference. No approximation, truncation, reset, or detached gradients.
    """
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    g = g.transpose(1, 2)
    lam = lam.transpose(1, 2)
    t = q.shape[-2]
    levels = level_indices(t, q.device)
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    weights = lam.gather(-1, levels.expand(*lam.shape[:2], t, t))
    gc = g.cumsum(-1)
    # Mask BEFORE exp: future differences can overflow for strongly negative g.
    decay = (gc[..., :, None] - gc[..., None, :]).masked_fill(~causal, -torch.inf).exp()
    scores = (q @ k.transpose(-1, -2)).tril()
    if beta is not None:
        beta = beta.transpose(1, 2)
        eye = torch.eye(t, device=q.device, dtype=q.dtype).expand_as(scores)
        system = (beta[..., :, None] * (k @ k.transpose(-1, -2))).tril(-1) + eye
        # scores @ inverse(system), evaluated as a right triangular solve.
        scores = torch.linalg.solve_triangular(system, scores, upper=False,
                                                left=False, unitriangular=True)
        v = beta[..., None] * v
    return ((scores * decay * weights) @ v).transpose(1, 2).contiguous()


class LogLinearMixer(nn.Module):
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
        self.l_proj = nn.Linear(d_model, self.heads * NUM_LEVELS, bias=False)
        self.L = nn.Parameter(torch.ones(self.heads, NUM_LEVELS))

    def forward(self, u, **kwargs):
        b, t, _ = u.shape
        if t > 256:
            raise ValueError('This MQAR implementation is configured for T <= 256')
        lam = F.softplus(self.l_proj(u).reshape(b, t, self.heads, NUM_LEVELS) * self.L)
        m = self.backbone
        if self.family == 'gdn':
            streams = []
            for proj, conv in ((m.q_proj,m.q_conv1d), (m.k_proj,m.k_conv1d), (m.v_proj,m.v_conv1d)):
                x, _ = conv(x=proj(u), cache=None, output_final_state=False, cu_seqlens=None)
                streams.append(x.reshape(b,t,self.heads,16))
            q, k, v = streams
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
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

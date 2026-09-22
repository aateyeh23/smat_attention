"""One-write/four-read SMAT with GDN boundary-transported keys and queries."""
import torch
from torch.nn import functional as F
from einops import rearrange
from fla.modules.l2norm import l2norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from zoo_smat_gdn import SmatGDNReset
from smat_gdn_transport import boundary_transport


class SmatGDNTransport(SmatGDNReset):
    def __init__(self, *args, write_hash_neighbor_grad=False,
                 transport_scalar_decay=True, detach_write_hash_input=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_hash_neighbor_grad = bool(write_hash_neighbor_grad)
        self.transport_scalar_decay = bool(transport_scalar_decay)
        self.detach_write_hash_input = bool(detach_write_hash_input)
        if self.d < 2 or not self.reset or self.memory_read_k != 4:
            raise ValueError('This experiment requires reset SMAT with exactly four reads')
        if self.memory_hash_source != 'hidden' or self.hash_key_shift is not None:
            raise ValueError('This experiment retains the learned causal hidden-state hash')
        # Preserve the original module/initialization ordering, but the transport
        # uses GDN Q/K rather than this independent memory feature projection.
        if hasattr(self, 'memory_qk'):
            self.memory_qk.requires_grad_(False)
        self.memory_update = 'transport_additive'

    def forward(self, u, **_):
        b, length, _ = u.shape
        g = self.gdn
        spec = self._spec(length, u.device)
        if spec is None or 2*spec.n != length:
            raise ValueError('Unsupported transport geometry')
        n = spec.n
        segment = u.reshape(b*2, n, -1)
        q, _ = g.q_conv1d(x=g.q_proj(segment), cache=None, output_final_state=False, cu_seqlens=None)
        k, _ = g.k_conv1d(x=g.k_proj(segment), cache=None, output_final_state=False, cu_seqlens=None)
        v, _ = g.v_conv1d(x=g.v_proj(segment), cache=None, output_final_state=False, cu_seqlens=None)
        q = rearrange(q, 'b t (h r) -> b t h r', h=self.h)
        k = rearrange(k, 'b t (h r) -> b t h r', h=self.h)
        v = rearrange(v, 'b t (h p) -> b t h p', h=self.h)
        raw_beta, raw_g = g.b_proj(segment), g.a_proj(segment)
        o, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=raw_g, beta=raw_beta, A_log=g.A_log, dt_bias=g.dt_bias,
            initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=g.allow_neg_eigval, state_v_first=True)
        o = o.reshape(b, length, self.h, self.p)
        if not self.disable_g:
            with torch.autocast('cuda', enabled=False):
                qn = l2norm(q.float()).reshape(b, length, self.h, self.r)
                kn = l2norm(k.float()).reshape(b, length, self.h, self.r)
                beta = raw_beta.float().sigmoid().reshape(b, length, self.h)
                if g.allow_neg_eigval:
                    beta = 2*beta
                log_g = (-g.A_log.float().exp()*F.softplus(raw_g.float()+g.dt_bias.float())).reshape(b,length,self.h)
                if not self.transport_scalar_decay:
                    log_g = torch.zeros_like(log_g)
                far_k, recent_q = boundary_transport(qn, kn, beta, log_g, n)
                memory_q = torch.cat((torch.zeros_like(qn[:, :n]), recent_q*self.r**-.5), dim=1)
                memory_k = torch.cat((far_k, torch.zeros_like(kn[:, n:])), dim=1)
                memory_v = v.float().reshape(b,length,self.h,self.p)
                hash_input = u.detach() if self.detach_write_hash_input else u
                key_source = self.kconv(F.pad(hash_input.float().transpose(1,2),(3,0))).transpose(1,2)
                ca = next(iter(self.ca[str(length)].values()))
                ca.delta_updates = False
                ca.anneal, ca.hard_k1 = 1., True
                ca.write_hash_neighbor_grad = self.write_hash_neighbor_grad
                if ca.read_k != 4:
                    raise ValueError('Four-read selector was not prebuilt')
                out = ca(u.float(), rearrange(memory_q,'b t h r -> (b h) t r'),
                         rearrange(memory_k,'b t h r -> (b h) t r'),
                         rearrange(memory_v,'b t h p -> (b h) t p'), n, key_src=key_source,
                         key_w=rearrange(beta[:,:n],'b t h -> (b h) t'))
                out = rearrange(out,'(b h) t p -> b t h p',b=b,h=self.h)
                lam = torch.sigmoid(self.alpha+self.lam_w(u.float())[:,n:])
                o = torch.cat((o[:,:n],o[:,n:]+(lam[...,None]*out).to(o.dtype)),dim=1)
        gate = rearrange(g.g_proj(u),'b t (h p) -> b t h p',h=self.h)
        return g.o_proj(rearrange(g.o_norm(o,gate),'b t h p -> b t (h p)'))

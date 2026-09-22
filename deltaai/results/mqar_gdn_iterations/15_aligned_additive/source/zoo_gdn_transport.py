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
                 transport_scalar_decay=True, detach_write_hash_input=False,
                 memory_plant=True, memory_read_address=False,
                 share_hash_across_lengths=False, address_read_sigma=.75,
                 transport_tied_features=False, memory_incidence_rescale=False,
                 orthogonal_hash=False, memory_hash_features='hidden',
                 symmetric_write_grad=False, memory_read_noise=0.,
                 joint_hash_balance=False, transport_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_hash_neighbor_grad = bool(write_hash_neighbor_grad)
        self.transport_scalar_decay = bool(transport_scalar_decay)
        self.detach_write_hash_input = bool(detach_write_hash_input)
        self.memory_plant = bool(memory_plant)
        self.memory_read_address = bool(memory_read_address)
        self.share_hash_across_lengths = bool(share_hash_across_lengths)
        self.transport_tied_features = bool(transport_tied_features)
        self.memory_incidence_rescale = bool(memory_incidence_rescale)
        self.orthogonal_hash = bool(orthogonal_hash)
        self.symmetric_write_grad = bool(symmetric_write_grad)
        self.memory_read_noise = float(memory_read_noise)
        self.joint_hash_balance = bool(joint_hash_balance)
        self.transport_enabled = bool(transport_enabled)
        if self.memory_read_noise < 0:
            raise ValueError('Read exploration noise must be nonnegative')
        if self.symmetric_write_grad and not self.write_hash_neighbor_grad:
            raise ValueError('Symmetric write gradients need the neighbor gradient adapter')
        if memory_hash_features not in ('hidden', 'gdn_keys'):
            raise ValueError(memory_hash_features)
        self.memory_hash_features = memory_hash_features
        if self.memory_hash_features == 'gdn_keys' and not self.transport_tied_features:
            self.kconv.requires_grad_(False)
        if self.orthogonal_hash and self.share_hash_across_lengths:
            raise ValueError('This experiment keeps separate orthogonal hashes per length')
        if address_read_sigma <= 0:
            raise ValueError('Address smoothing must be positive')
        shared_hash = None
        for modules in self.ca.values():
            for ca in modules.values():
                ca.plant = self.memory_plant
                ca.address_reads = self.memory_read_address
                ca.symmetric_write_grad = self.symmetric_write_grad
                ca.joint_hash_balance = self.joint_hash_balance
                ca.address_read_sigma = float(address_read_sigma)
                if self.orthogonal_hash:
                    from torch.nn.utils.parametrizations import orthogonal
                    # Cayley coordinates stay valid under AdamW decay. The
                    # Householder parametrization stores integer sign slots
                    # inside its parameter, which this optimizer would decay.
                    with torch.random.fork_rng(devices=[]):
                        orthogonal(ca, 'W', orthogonal_map='cayley')
                if self.share_hash_across_lengths:
                    if shared_hash is None:
                        shared_hash = ca
                    else:
                        ca.W, ca.gamma, ca.b = shared_hash.W, shared_hash.gamma, shared_hash.b
                if self.memory_read_address:
                    ca.W_read.requires_grad_(False)
                    ca.b_read.requires_grad_(False)
        if self.d < 2 or not self.reset or self.memory_read_k != 4:
            raise ValueError('This experiment requires reset SMAT with exactly four reads')
        if self.memory_hash_source != 'hidden' or self.hash_key_shift is not None:
            raise ValueError('This experiment retains the learned causal hidden-state hash')
        # Preserve the original module/initialization ordering, but the transport
        # uses GDN Q/K rather than this independent memory feature projection.
        if hasattr(self, 'memory_qk'):
            self.memory_qk.requires_grad_(self.transport_tied_features)
        elif self.transport_tied_features:
            raise ValueError('Tied transport requires the causal memory projection')
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
                if self.transport_tied_features:
                    # The same learned causal context supplies writer addresses
                    # and retrieval keys. Content loss can now train the causal
                    # convolution directly, in addition to the routing surrogate.
                    # Reset the short convolution at the existing boundary.
                    content_source = self.kconv(F.pad(segment.float().transpose(1,2),(3,0)))
                    content_source = content_source.transpose(1,2).reshape(b,length,-1)
                    qn = l2norm(self.memory_qk(u.float()).reshape(b,length,self.h,self.r))
                    kn = l2norm(self.memory_qk(content_source).reshape(b,length,self.h,self.r))
                else:
                    qn = l2norm(q.float()).reshape(b, length, self.h, self.r)
                    kn = l2norm(k.float()).reshape(b, length, self.h, self.r)
                beta = raw_beta.float().sigmoid().reshape(b, length, self.h)
                if g.allow_neg_eigval:
                    beta = 2*beta
                log_g = (-g.A_log.float().exp()*F.softplus(raw_g.float()+g.dt_bias.float())).reshape(b,length,self.h)
                if not self.transport_scalar_decay:
                    log_g = torch.zeros_like(log_g)
                if self.transport_enabled:
                    far_k, recent_q = boundary_transport(qn, kn, beta, log_g, n)
                else:
                    far_k, recent_q = kn[:,:n] * beta[:,:n,...,None], qn[:,n:]
                memory_q = torch.cat((torch.zeros_like(qn[:, :n]), recent_q*self.r**-.5), dim=1)
                memory_k = torch.cat((far_k, torch.zeros_like(kn[:, n:])), dim=1)
                memory_v = v.float().reshape(b,length,self.h,self.p)
                if self.memory_hash_features == 'gdn_keys':
                    # Reuse the content keys trained by local GDN and transport.
                    # Back-project to model width with the existing projection;
                    # this also works when concatenated key width differs.
                    projection = self.memory_qk if self.transport_tied_features else g.k_proj
                    key_source = F.linear(kn.flatten(2), projection.weight.T)
                    if self.detach_write_hash_input:
                        key_source = key_source.detach()
                else:
                    hash_input = u.detach() if self.detach_write_hash_input else u
                    key_source = self.kconv(F.pad(hash_input.float().transpose(1,2),(3,0))).transpose(1,2)
                ca = next(iter(self.ca[str(length)].values()))
                if self.training:
                    self._steps += 1
                ca.read_noise_std = self.memory_read_noise * max(
                    0., 1. - self._steps / max(1, self.anneal_steps))
                ca.delta_updates = False
                ca.anneal, ca.hard_k1 = 1., True
                ca.write_hash_neighbor_grad = self.write_hash_neighbor_grad
                if ca.read_k != 4:
                    raise ValueError('Four-read selector was not prebuilt')
                # In additive mode key_w only weights the hash-balance loss;
                # the actual write beta is already carried by far_k.
                hash_write_weight = beta[:,:n].detach() if self.detach_write_hash_input else beta[:,:n]
                out = ca(u.float(), rearrange(memory_q,'b t h r -> (b h) t r'),
                         rearrange(memory_k,'b t h r -> (b h) t r'),
                         rearrange(memory_v,'b t h p -> (b h) t p'), n, key_src=key_source,
                         key_w=rearrange(hash_write_weight,'b t h -> (b h) t'))
                out = rearrange(out,'(b h) t p -> b t h p',b=b,h=self.h)
                if self.memory_incidence_rescale:
                    # A uniformly selected plane contains any bucket with
                    # probability 1/n_cosets. Correct this geometry-dependent
                    # average attenuation before the learned memory gate.
                    out = out * (ca.n_cosets if ca.mode == 'plane' else ca.N0)
                lam = torch.sigmoid(self.alpha+self.lam_w(u.float())[:,n:])
                o = torch.cat((o[:,:n],o[:,n:]+(lam[...,None]*out).to(o.dtype)),dim=1)
        gate = rearrange(g.g_proj(u),'b t (h p) -> b t h p',h=self.h)
        return g.o_proj(rearrange(g.o_norm(o,gate),'b t h p -> b t (h p)'))

"""Incremental decoding for the frozen PG19 GDN/Mamba-2 and SMAT models.

Prefill uses the unmodified fixed-window model and retains its SMAT summaries.
Decode updates only convolution buffers, recurrent states, and (for SMAT) the
boundary-to-query transport. No prior hidden sequence is kept or recomputed.
SMAT prompts must already include the trained landmark half; generation cannot
extend beyond the trained window or silently change its hash geometry.
"""
import torch
from torch.nn import functional as F
from einops import rearrange
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.modules.l2norm import l2norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from zoology.mixers.mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def conv_buffer(projected, lengths, width, boundary=0):
    """Raw convolution inputs at each sequence's last actual (unpadded) token."""
    buffers = []
    for row, length in enumerate(lengths):
        begin = max(boundary, length-width)
        x = projected[row, begin:length].transpose(0, 1)
        buffers.append(F.pad(x, (width-x.shape[-1], 0)))
    return torch.stack(buffers).contiguous()


class RecurrentDecoder:
    def __init__(self, model):
        if model.training:
            raise ValueError('Cached decoding requires model.eval()')
        self.model = model
        self.length = model.cfg.length
        self.smat = model.cfg.d > 1
        self.boundary = self.length//2 if self.smat else 0
        self.caches = []
        self.lengths = None

    @torch.inference_mode()
    def prefill(self, input_ids):
        """Return first next-token logits and initialize an independent cache."""
        lengths = [len(row) for row in input_ids]
        if not lengths or min(lengths) <= self.boundary or max(lengths) > self.length:
            raise ValueError('Prompts must fit the trained window and, for SMAT, extend past its midpoint')
        device = self.model.embedding.weight.device
        tokens = torch.full((len(lengths), self.length), self.model.cfg.vocab-1,
                            device=device, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            tokens[i, :len(ids)] = torch.as_tensor(ids, device=device)
        self.lengths = lengths
        self.caches = [dict() for _ in self.model.blocks]
        import smat_read_triton
        original_read = smat_read_triton.read_topk
        active = [None]
        def capture_memory(q, indices, weights, memory):
            if active[0] is None:
                raise RuntimeError('SMAT read outside the active prefill layer')
            self.caches[active[0]]['memory'] = memory
            return original_read(q, indices, weights, memory)
        handles = []
        for index, block in enumerate(self.model.blocks):
            def before(module, args, index=index):
                active[0] = index
            def after(module, args, output, index=index):
                cache = self.caches[index]
                if hasattr(module, 'gdn'):
                    self._prefill_gdn(module, args[0], cache)
                else:
                    self._prefill_mamba(module, args[0], cache)
            handles.extend([block.mixer.register_forward_pre_hook(before),
                            block.mixer.register_forward_hook(after)])
        smat_read_triton.read_topk = capture_memory
        try:
            hidden, _ = self.model.hidden(tokens)
        finally:
            smat_read_triton.read_topk = original_read
            for handle in handles:
                handle.remove()
        if self.smat and any('memory' not in cache for cache in self.caches):
            raise RuntimeError('Not every layer produced a cached SMAT memory')
        positions = torch.tensor(lengths, device=device)-1
        return F.linear(hidden[torch.arange(len(lengths), device=device), positions],
                        self.model.embedding.weight).float()

    def _prefill_gdn(self, mixer, u, cache):
        g = mixer.gdn
        b, length, _ = u.shape
        segment = u.reshape(b*2, length//2, -1) if self.smat else u
        values = []
        for name in ('q','k','v'):
            conv = getattr(g, name+'_conv1d')
            raw = getattr(g, name+'_proj')(segment)
            y, _ = conv(x=raw, cache=None, output_final_state=False, cu_seqlens=None)
            raw = raw.reshape(b, length, -1)
            cache[name+'_conv'] = conv_buffer(raw, self.lengths, conv.kernel_size[0], self.boundary)
            values.append(y.reshape(b, length, mixer.h, -1))
        q, k, v = values
        beta = g.b_proj(segment).reshape(b, length, mixer.h)
        gate = g.a_proj(segment).reshape(b, length, mixer.h)
        # Neutral padded transitions let one fixed-shape scan recover all
        # variable-length prompt states without per-length kernel compilation.
        valid = torch.arange(length-self.boundary,device=u.device)[None,:] < (
            torch.tensor(self.lengths,device=u.device)[:,None]-self.boundary)
        qs,ks,vs = [t[:,self.boundary:].contiguous() for t in (q,k,v)]
        raw_beta = beta[:,self.boundary:].masked_fill(~valid[:,:,None],float('-inf'))
        raw_gate = gate[:,self.boundary:].masked_fill(~valid[:,:,None],float('-inf'))
        _,cache['state'] = chunk_gated_delta_rule(q=qs,k=ks,v=vs,g=raw_gate,beta=raw_beta,
            A_log=g.A_log,dt_bias=g.dt_bias,initial_state=None,output_final_state=True,
            use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,allow_neg_eigval=g.allow_neg_eigval,state_v_first=True)
        if self.smat:
            assert mixer.transport_enabled and mixer.transport_scalar_decay and not mixer.transport_tied_features
            assert not mixer.memory_incidence_rescale and not mixer.profile_delta
            with torch.autocast('cuda',enabled=False):
                qn,kn=l2norm(qs.float()),l2norm(ks.float())
                bet=raw_beta.float().sigmoid()*(2 if g.allow_neg_eigval else 1)
                log_g=-g.A_log.float().exp()*F.softplus(raw_gate.float()+g.dt_bias.float())
                identity=torch.eye(mixer.r,device=u.device).expand(b,mixer.h,mixer.r,mixer.r).contiguous()
                _,cache['transport']=chunk_gated_delta_rule(q=qn,k=kn,v=torch.zeros_like(qn),
                    beta=bet,g=log_g,initial_state=identity,output_final_state=True,
                    scale=1.,use_qk_l2norm_in_kernel=False,state_v_first=True,chunk_size=16)
            cache['ca'] = next(iter(mixer.ca[str(self.length)].values()))

    def _prefill_mamba(self, mixer, u, cache):
        m = mixer.mixer
        b, length, _ = u.shape
        assert m.d_ssm == m.d_inner and m.ngroups == 1 and m.dt_limit == (0., float('inf'))
        packed = m.in_proj(u)
        _, raw, dt = packed.split([m.d_ssm, m.d_ssm+2*m.d_state, m.nheads], dim=-1)
        cache['conv'] = conv_buffer(raw, self.lengths, m.d_conv, self.boundary)
        if self.smat:
            conv_input = raw.reshape(b*2, length//2, -1).transpose(1,2).contiguous()
            convolved = causal_conv1d_fn(conv_input, m.conv1d.weight[:,0], m.conv1d.bias,
                                         activation=m.activation).transpose(1,2).reshape(b,length,-1)
        else:
            convolved = m.act(m.conv1d(raw.transpose(1,2)).transpose(1,2))[:, :length]
        x, key, query = convolved.split([m.d_ssm,m.d_state,m.d_state], dim=-1)
        x = x.reshape(b,length,m.nheads,m.headdim)
        key, query = key.unsqueeze(2), query.unsqueeze(2)
        A = -m.A_log.float().exp()
        valid=torch.arange(length-self.boundary,device=u.device)[None,:] < (
            torch.tensor(self.lengths,device=u.device)[:,None]-self.boundary)
        masked_dt=dt[:,self.boundary:].masked_fill(~valid[:,:,None],float('-inf'))
        _,state=mamba_chunk_scan_combined(x[:,self.boundary:].contiguous(),masked_dt,A,
            key[:,self.boundary:].contiguous(),query[:,self.boundary:].contiguous(),
            chunk_size=m.chunk_size,D=m.D,dt_bias=m.dt_bias,dt_softplus=True,return_final_states=True)
        cache['state']=state.float().contiguous()
        cache['A'] = A[:,None,None].expand(m.nheads,m.headdim,m.d_state)
        cache['dt_bias'] = m.dt_bias[:,None].expand(m.nheads,m.headdim)
        cache['D'] = m.D[:,None].expand(m.nheads,m.headdim)
        if self.smat:
            assert mixer.kernel == 'id' and mixer.g_decay and mixer.lam_act == 'sigmoid'
            with torch.autocast('cuda', enabled=False):
                increments = A[None,None,:]*F.softplus(dt.float()+m.dt_bias.float())
                cumulative = increments.cumsum(dim=1)
                cache['log_total'] = torch.stack([cumulative[i,end-1] for i,end in enumerate(self.lengths)])
                cache['log_boundary'] = cumulative[:,self.boundary-1].clone()
            cache['ca'] = next(iter(mixer.ca[str(self.length)].values()))

    @torch.inference_mode()
    def step(self, token_ids):
        if self.lengths is None or max(self.lengths) >= self.length:
            raise ValueError('Prefill first and reserve room inside the trained window')
        token_ids = torch.as_tensor(token_ids, device=self.model.embedding.weight.device).reshape(-1,1)
        if len(token_ids) != len(self.lengths):
            raise ValueError('Decode batch size differs from prefill')
        x = self.model.embedding(token_ids)
        for block, cache in zip(self.model.blocks, self.caches):
            u = block.norm(x)
            if hasattr(block.mixer, 'gdn'):
                y = self._step_gdn(block.mixer,u,cache)
            else:
                y = self._step_mamba(block.mixer,u,cache)
            x = x+y
            gate, value = block.up_gate(block.norm2(x)).chunk(2,dim=-1)
            x = x+block.down(F.silu(gate)*value)
        self.lengths = [length+1 for length in self.lengths]
        return F.linear(self.model.norm(x)[:,0],self.model.embedding.weight).float()

    def _read(self, u, query, cache):
        import smat_read_triton
        indices, weights = cache['ca']._topk_reads(u.float())
        return smat_read_triton.read_topk(query,indices,weights.to(query.dtype),cache['memory'])

    def _step_gdn(self, mixer, u, cache):
        g, b = mixer.gdn, u.shape[0]
        features = []
        for name in ('q','k','v'):
            y, state = getattr(g,name+'_conv1d')(x=getattr(g,name+'_proj')(u),
                cache=cache[name+'_conv'],output_final_state=True,cu_seqlens=None)
            cache[name+'_conv'] = state
            features.append(y.reshape(b,1,mixer.h,-1))
        q,k,v = features
        beta, gate = g.b_proj(u), g.a_proj(u)
        o, cache['state'] = fused_recurrent_gated_delta_rule(q=q,k=k,v=v,g=gate,beta=beta,
            A_log=g.A_log,dt_bias=g.dt_bias,initial_state=cache['state'],output_final_state=True,
            use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=g.allow_neg_eigval,state_v_first=True)
        if self.smat:
            with torch.autocast('cuda',enabled=False):
                qn,kn = l2norm(q.float()),l2norm(k.float())
                bet = beta.float().sigmoid()*(2 if g.allow_neg_eigval else 1)
                log_g = -g.A_log.float().exp()*F.softplus(gate.float()+g.dt_bias.float())
                transported,cache['transport'] = fused_recurrent_gated_delta_rule(
                    q=qn,k=kn,v=torch.zeros_like(qn),g=log_g,beta=bet,scale=1.,
                    initial_state=cache['transport'],output_final_state=True,state_v_first=True)
                query = rearrange(transported*mixer.r**-.5,'b t h r -> (b h) t r')
                memory_out = self._read(u,query,cache).reshape(b,mixer.h,1,mixer.p).transpose(1,2)
                lam = torch.sigmoid(mixer.alpha+mixer.lam_w(u.float()))
                o = o+(lam[...,None]*memory_out).to(o.dtype)
        gate = g.g_proj(u).reshape(b,1,mixer.h,mixer.p)
        return g.o_proj(g.o_norm(o,gate).flatten(2))

    def _step_mamba(self, mixer, u, cache):
        m, b = mixer.mixer,u.shape[0]
        packed = m.in_proj(u)[:,0]
        z,raw,dt = packed.split([m.d_ssm,m.d_ssm+2*m.d_state,m.nheads],dim=-1)
        convolved = causal_conv1d_update(raw,cache['conv'],m.conv1d.weight[:,0],m.conv1d.bias,
                                        activation=m.activation if self.smat else None)
        if not self.smat:
            convolved = F.silu(convolved)
        x,key,query = convolved.split([m.d_ssm,m.d_state,m.d_state],dim=-1)
        y = selective_state_update(cache['state'],x.reshape(b,m.nheads,m.headdim),
            dt[:,:,None].expand(b,m.nheads,m.headdim),cache['A'],key[:,None,:],query[:,None,:],
            cache['D'],dt_bias=cache['dt_bias'],dt_softplus=True)
        if self.smat:
            with torch.autocast('cuda',enabled=False):
                cache['log_total'] += -m.A_log.float().exp()[None,:]*F.softplus(dt.float()+m.dt_bias.float())
                decay = (cache['log_total']-cache['log_boundary']).exp()
                phi = query.float()[:,None,None,:].expand(b,m.nheads,1,m.d_state).reshape(b*m.nheads,1,m.d_state)
                memory_out = self._read(u,phi,cache).reshape(b,m.nheads,m.headdim)
                lam = torch.sigmoid(mixer.alpha[:,0][None,:]+mixer.lam_w(u.float())[:,0])
                y = y+(lam[:,:,None]*decay[:,:,None]*memory_out).to(y.dtype)
        y = y.reshape(b,1,m.d_ssm)
        return m.out_proj(m.norm(y,z[:,None,:]))

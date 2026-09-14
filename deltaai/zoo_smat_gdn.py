"""GDN on causal segments, with the current content-hashed SMAT cross-boundary memory.

G uses additive beta*k*v writes and sparse hyperplane reads. It has no extra
temporal decay; GDN's gated delta updates apply within each causal segment.
"""
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from zoo_smat_mixer import SmatMamba2MR


class SmatGDNReset(nn.Module):
    _spec = SmatMamba2MR._spec
    get_auxiliary_loss = SmatMamba2MR.get_auxiliary_loss

    def __init__(self, d_model, layer_idx=0, d=3, headdim=16, expand_v=1,
                 n_heads=None, reset=True, lam_bias=-2.197224577,
                 anneal_steps=1000, balance_coef=0.01, hash_codim=1,
                 prebuild_lengths=(64, 128, 256), memory_update="additive", hash_key_shift=None,
                 memory_hash_source="hidden", memory_key_mode="base", memory_route_full_grad=False, memory_route_soft=False, **_):
        super().__init__()
        if memory_update not in ("additive", "delta"):
            raise ValueError(memory_update)
        self.memory_update = memory_update
        if hash_key_shift not in (None, 1):
            raise ValueError('hash_key_shift must be None or 1')
        self.hash_key_shift = hash_key_shift
        if memory_hash_source not in ('hidden','gdn'):
            raise ValueError(memory_hash_source)
        self.memory_hash_source = memory_hash_source
        if memory_key_mode not in ('base','tied_previous','tied_causal'):
            raise ValueError(memory_key_mode)
        self.memory_key_mode = memory_key_mode
        self.memory_route_full_grad = memory_route_full_grad
        self.memory_route_soft = memory_route_soft
        self.gdn = GatedDeltaNet(hidden_size=d_model, head_dim=headdim,
            num_heads=n_heads if n_heads is not None else (1 if d_model == 16 else 2), expand_v=expand_v,
            mode="chunk", use_gate=True, use_short_conv=True, layer_idx=layer_idx)
        g = self.gdn
        self.d, self.reset, self.n_P = d, reset, 0
        self.h, self.r, self.p = g.num_heads, g.head_k_dim, g.head_v_dim
        self.read, self.hash_src, self.hash_dim = "chash", "hidden", d_model
        self.hash_mode, self.hash_codim = "point", hash_codim
        self.hash_conv, self.hash_shift, self.hash_freeze = True, 0, False
        self.hash_lr_scale, self.sparse_ops = 1.0, True
        self.specs, self.ca = {}, nn.ModuleDict()
        self.anneal_steps, self.balance_coef, self._steps = anneal_steps, balance_coef, 0
        self.disable_g = False  # diagnostic only: preserves the recurrence reset
        if d >= 2:
            self.kconv = nn.Conv1d(d_model, d_model, 4, groups=d_model, bias=False)
            nn.init.constant_(self.kconv.weight, 0.25)
            self.lam_w = nn.Linear(d_model, self.h)
            nn.init.zeros_(self.lam_w.weight); nn.init.zeros_(self.lam_w.bias)
            self.alpha = nn.Parameter(torch.full((1, 1, self.h), float(lam_bias)))
            # Register every mixture length's hash before the optimizer is built.
            for length in prebuild_lengths:
                self._spec(length, torch.device("cpu"))
            self.specs.clear()  # device-specific geometry is rebuilt after .to(cuda)
            if memory_key_mode in ('tied_previous','tied_causal'):
                # Preserve the initialization RNG stream for the unchanged layers.
                with torch.random.fork_rng(devices=[]):
                    self.memory_qk = nn.Linear(d_model, self.h*self.r, bias=False)
                with torch.no_grad(): self.memory_qk.weight.copy_(g.q_proj.weight)

    def forward(self, u, **_):
        b, length, _ = u.shape
        g = self.gdn
        spec = self._spec(length, u.device) if self.d >= 2 else None
        fold = self.reset and spec is not None
        n = spec.n if spec is not None else length // 2
        if fold:
            assert 2 * n == length
        segment = u.reshape(b * 2, n, -1) if fold else u
        q, _ = g.q_conv1d(x=g.q_proj(segment), cache=None, output_final_state=False, cu_seqlens=None)
        k, _ = g.k_conv1d(x=g.k_proj(segment), cache=None, output_final_state=False, cu_seqlens=None)
        v, _ = g.v_conv1d(x=g.v_proj(segment), cache=None, output_final_state=False, cu_seqlens=None)
        q = rearrange(q, "b t (h r) -> b t h r", h=self.h)
        k = rearrange(k, "b t (h r) -> b t h r", h=self.h)
        v = rearrange(v, "b t (h p) -> b t h p", h=self.h)
        beta = g.b_proj(segment)
        o, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g.a_proj(segment), beta=beta,
            A_log=g.A_log, dt_bias=g.dt_bias, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, allow_neg_eigval=g.allow_neg_eigval,
            state_v_first=True)
        if fold:
            q, k, v, o = [t.reshape(b, length, self.h, -1) for t in (q, k, v, o)]
            beta = beta.reshape(b, length, self.h)
        hash_u = u
        if spec is not None and self.memory_hash_source == 'gdn':
            base_gate = rearrange(g.g_proj(u), 'b t (h p) -> b t h p', h=self.h)
            hash_u = g.o_proj(rearrange(g.o_norm(o, base_gate), 'b t h p -> b t (h p)'))
        if spec is not None and not self.disable_g:
            if self.training:
                self._steps += 1
                for mods in self.ca.values():
                    for ca in mods.values():
                        ca.anneal = min(1.0, self._steps / max(1, self.anneal_steps))
            with torch.autocast("cuda", enabled=False):
                qf = rearrange(F.normalize(q.float(), dim=-1, eps=1e-6), "b t h r -> (b h) t r")
                kf = rearrange(F.normalize(k.float(), dim=-1, eps=1e-6), "b t h r -> (b h) t r")
                if self.memory_key_mode == 'tied_previous':
                    projected = rearrange(self.memory_qk(u.float()), 'b t (h r) -> (b h) t r', h=self.h)
                    qf = F.normalize(projected, dim=-1, eps=1e-6)
                    kf = F.pad(qf[:, :-1], (0, 0, 1, 0))
                writes = beta.float().sigmoid()
                values = v.float() if self.memory_update == "delta" else v.float() * writes.unsqueeze(-1)
                vf = rearrange(values, "b t h p -> (b h) t p")
                key_source = self.kconv(F.pad(hash_u.float().transpose(1, 2), (3, 0))).transpose(1, 2)
                if self.hash_key_shift == 1:
                    key_source = F.pad(u.float()[:, :-1], (0, 0, 1, 0))
                if self.memory_key_mode == 'tied_causal':
                    # Learn which causal context to associate with each write.
                    qf = rearrange(self.memory_qk(hash_u.float()), 'b t (h r) -> (b h) t r', h=self.h)
                    kf = rearrange(self.memory_qk(key_source), 'b t (h r) -> (b h) t r', h=self.h)
                    qf = F.normalize(qf, dim=-1, eps=1e-6)
                    kf = F.normalize(kf, dim=-1, eps=1e-6)
                ca = next(iter(self.ca[str(length)].values()))
                ca.delta_updates = self.memory_update == "delta"
                if self.memory_route_soft:
                    ca.anneal = 0.0  # same sparse neighboring-cell mixture in train and eval
                if self.memory_route_full_grad:
                    ca.hard_k1 = not self.training  # retain zero-weight neighbors only for backward
                out = ca(hash_u.float(), qf, kf, vf, n, key_src=key_source,
                         key_w=rearrange(writes[:, :n], "b t h -> (b h) t"))
                out = rearrange(out, "(b h) t p -> b t h p", b=b, h=self.h)
                lam = torch.sigmoid(self.alpha + self.lam_w(u.float())[:, n:])
                o = torch.cat([o[:, :n], o[:, n:] + (lam.unsqueeze(-1) * out).to(o.dtype)], dim=1)
        gate = rearrange(g.g_proj(u), "b t (h p) -> b t h p", h=self.h)
        o = g.o_norm(o, gate)
        return g.o_proj(rearrange(o, "b t h p -> b t (h p)"))

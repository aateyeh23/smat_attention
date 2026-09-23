"""SMAT as a Zoology sequence mixer: Mamba-2 (mamba_ssm) on the causal blocks
with a state reset at the landmark/recent boundary, plus SMAT's landmark block G
(positive kernel, pooled by profile, incidence, type-major readout, own
normaliser, per-head mask-bank offsets) added to the recent rows.

Handles Zoology's mixed sequence lengths by building one mask per T (cached),
with chunk = T/4 and d_eff = min(d, d_max(T)); d_eff < 2 (or d == 1) means
plain Mamba-2 with no reset and no G.
"""
import math, os, sys
import torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "smat")):
    if p not in sys.path: sys.path.insert(0, p)
from smat_mask import build_mask, d_max_geometric
from smat_attn import to_device, featurise, make_phi, pool_profiles, apply_incidence, query_by_type
# Zoology's vendored Mamba-2 (the class its own mamba2 arm uses; use_mem_eff_path=False)
from zoology.mixers.mamba2 import Mamba2


class SmatMamba2(nn.Module):
    def __init__(self, d_model, layer_idx=0, d=2, n_heads=1, r=64, d_state=128, mask_bank=True,
                 qk_norm=True, n_P=0, reset=True, **_):
        super().__init__()
        # reset=False: the hybrid -- Mamba-2 runs over the whole sequence (recall is
        # Mamba-2's), G is purely additive routing on the recent rows.
        self.reset = reset
        d_inner = 2 * d_model
        self.mixer = Mamba2(d_model, d_state=d_state, headdim=min(64, d_inner), layer_idx=layer_idx)
        self.d, self.h, self.dh, self.n_P = d, n_heads, d_model // n_heads, n_P
        self.offsets = [layer_idx * n_heads + h for h in range(n_heads)] if mask_bank else None
        self.specs = {}
        if d >= 2:
            self.qn = nn.LayerNorm(self.dh) if qk_norm else nn.Identity()
            self.kn = nn.LayerNorm(self.dh) if qk_norm else nn.Identity()
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.out.weight)        # G starts switched off: the layer begins as pure Mamba-2
            self.r, self.phi = r, None

    def _spec(self, T, device):
        if T not in self.specs:
            chunk = max(16, T // 4)
            d_eff = min(self.d, d_max_geometric(T, chunk=chunk))
            self.specs[T] = to_device(build_mask(T, d_eff, chunk=chunk, n_P=self.n_P), device) if d_eff >= 2 else None
        return self.specs[T]

    def forward(self, x, **_):
        nb, T, C = x.shape
        spec = self._spec(T, x.device) if self.d >= 2 else None
        if spec is None:
            return self.mixer(x)
        n = spec.n
        if self.reset:
            # exact state + conv reset at the boundary without varlen kernels: the two
            # segments are equal halves (n = T/2), so fold them into the batch dimension
            assert 2 * n == T, f"reset by folding needs n = T/2 (n={n}, T={T})"
            y = self.mixer(x.reshape(nb * 2, n, C)).reshape(nb, T, C)
        else:
            y = self.mixer(x)
        if self.phi is None:                                 # fixed random projection, built on the right device once
            self.phi = make_phi(self.dh, self.r, device=x.device, dtype=torch.float32, seed=0)
        q, k, v = self.qkv(x).view(nb, T, 3, self.h, self.dh).unbind(2)
        q, k = self.qn(q), self.kn(k)
        with torch.autocast(device_type=x.device.type, enabled=False):
            fold = lambda t: t.transpose(1, 2).reshape(nb * self.h, T, self.dh).float()
            Phi, Psi, Vb = featurise(fold(q), fold(k), fold(v), self.phi)
            if spec.n_X > 0:
                Fp = pool_profiles(Psi, Vb, spec, acc_dtype=torch.float32, backend="torch")
                U = apply_incidence(Fp, spec, backend="torch")
            else:
                U = torch.zeros(nb * self.h, spec.B, Phi.shape[-1], Vb.shape[-1], device=x.device)
            if spec.n_P:
                U = U + (Psi[:, :spec.n_P].transpose(-1, -2) @ Vb[:, :spec.n_P]).unsqueeze(1)
            if self.offsets is not None:
                B = spec.B
                Uh = U.view(nb, self.h, B, U.shape[-2], U.shape[-1])
                Uh = torch.stack([torch.roll(Uh[:, i], shifts=-int(self.offsets[i]) % B, dims=1) for i in range(self.h)], dim=1)
                U = Uh.reshape(nb * self.h, B, U.shape[-2], U.shape[-1])
            Ylr = query_by_type(Phi[:, n:], U, spec)
            Ylr = Ylr[..., :-1] / Ylr[..., -1:].clamp_min(1e-6)
            Ylr = Ylr.view(nb, self.h, T - n, self.dh).transpose(1, 2).reshape(nb, T - n, C)
        return torch.cat([y[:, :n], y[:, n:] + self.out(Ylr.to(y.dtype))], dim=1)


# ---------------------------------------------------------------------------
# Multi-resolution form: Mamba-2 with SMAT's G as an extra level, sharing
# Mamba-2's own B / C / x stream.  Exactly Mamba-2 when the per-head amplitude
# alpha == 0 (its init), so Mamba-2 is a point in the parameter space.
#   level 0 : the (un-reset) SSD state           -- Mamba-2's own recurrence
#   level d : SMAT's hyperplane block G, keys phi(B_t), queries phi(C_t), values x_t
#   y_t    += lambda_t * G_t,   lambda_t = alpha_h * sigmoid(w . u_t)
# G's contribution is added before Mamba-2's gated RMSNorm and out_proj.
# ---------------------------------------------------------------------------
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from causal_conv1d import causal_conv1d_fn


class SmatMamba2Shared(nn.Module):
    def __init__(self, d_model, layer_idx=0, d=2, r=64, d_state=128, mask_bank=True, n_P=0,
                 reset=False, lam_bias=-8.0, detach_g=True, **_):
        super().__init__()
        d_inner = 2 * d_model
        # lam_bias: lambda_t = sigmoid(alpha_h + w.u_t), alpha init lam_bias (-8 -> 3e-4: Mamba-2 to ~1e-4)
        # detach_g: G reads B, C, x with stop-gradient, so it cannot alter Mamba-2's own training
        self.lam_bias, self.detach_g = lam_bias, detach_g
        self.mixer = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                            headdim=min(64, d_inner), ngroups=1, chunk_size=64, use_mem_eff_path=False)
        m = self.mixer
        assert m.d_ssm == m.d_inner, "d_mlp path not supported"
        self.d, self.n_P, self.reset, self.r = d, n_P, reset, r
        self.h, self.p, self.N = m.nheads, m.headdim, m.d_state
        self.offsets = [layer_idx * self.h + i for i in range(self.h)] if mask_bank else None
        self.specs, self.phi = {}, None
        if d >= 2:
            self.lam_w = nn.Linear(d_model, self.h)
            nn.init.zeros_(self.lam_w.weight); nn.init.zeros_(self.lam_w.bias)
            self.alpha = nn.Parameter(torch.full((self.h,), float(lam_bias)))   # lambda ~ sigmoid(lam_bias) at init

    def _spec(self, T, device):
        if T not in self.specs:
            chunk = max(16, T // 4)
            d_eff = min(self.d, d_max_geometric(T, chunk=chunk))
            self.specs[T] = to_device(build_mask(T, d_eff, chunk=chunk, n_P=self.n_P), device) if d_eff >= 2 else None
        return self.specs[T]

    def forward(self, u, **_):
        m = self.mixer
        b, l, _ = u.shape
        spec = self._spec(l, u.device) if self.d >= 2 else None
        seq_idx = None
        if spec is not None and self.reset:
            seq_idx = torch.zeros(b, l, dtype=torch.int32, device=u.device); seq_idx[:, spec.n:] = 1
        zxbcdt = m.in_proj(u)
        A = -torch.exp(m.A_log.float())
        z, xBC, dt = torch.split(zxbcdt, [m.d_ssm, m.d_ssm + 2 * m.ngroups * m.d_state, m.nheads], dim=-1)
        xBC = causal_conv1d_fn(xBC.contiguous().transpose(1, 2), rearrange(m.conv1d.weight, "d 1 w -> d w"),
                               bias=m.conv1d.bias, activation=m.activation, seq_idx=seq_idx).transpose(1, 2)
        x, B, C = torch.split(xBC, [m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state], dim=-1)
        dt_limit_kwargs = {} if m.dt_limit == (0.0, float("inf")) else dict(dt_limit=m.dt_limit)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=m.headdim), dt, A,
            rearrange(B, "b l (g n) -> b l g n", g=m.ngroups), rearrange(C, "b l (g n) -> b l g n", g=m.ngroups),
            chunk_size=m.chunk_size, D=m.D, z=None, dt_bias=m.dt_bias, dt_softplus=True,
            seq_idx=seq_idx, **dt_limit_kwargs)                                   # (b, l, h, p)
        if spec is not None:
            n = spec.n
            if self.phi is None:
                self.phi = make_phi(self.N, self.r, device=u.device, dtype=torch.float32, seed=0)
            with torch.autocast(device_type=u.device.type, enabled=False):
                Bg, Cg, xg = (B.detach(), C.detach(), x.detach()) if self.detach_g else (B, C, x)
                Bf, Cf = Bg.float().reshape(b, l, m.ngroups * m.d_state), Cg.float().reshape(b, l, m.ngroups * m.d_state)
                Phi = self.phi(Cf).unsqueeze(1).expand(b, self.h, l, self.r).reshape(b * self.h, l, self.r)
                Psi = self.phi(Bf).unsqueeze(1).expand(b, self.h, l, self.r).reshape(b * self.h, l, self.r)
                xv = rearrange(xg.float(), "b l (h p) -> (b h) l p", p=self.p)
                Vb = torch.cat([xv, torch.ones_like(xv[..., :1])], dim=-1)
                if spec.n_X > 0:
                    Fp = pool_profiles(Psi, Vb, spec, acc_dtype=torch.float32, backend="torch")
                    U = apply_incidence(Fp, spec, backend="torch")
                else:
                    U = torch.zeros(b * self.h, spec.B, self.r, self.p + 1, device=u.device)
                if spec.n_P:
                    U = U + (Psi[:, :spec.n_P].transpose(-1, -2) @ Vb[:, :spec.n_P]).unsqueeze(1)
                if self.offsets is not None:
                    Bn = spec.B
                    Uh = U.view(b, self.h, Bn, self.r, self.p + 1)
                    Uh = torch.stack([torch.roll(Uh[:, i], shifts=-int(self.offsets[i]) % Bn, dims=1) for i in range(self.h)], dim=1)
                    U = Uh.reshape(b * self.h, Bn, self.r, self.p + 1)
                Ylr = query_by_type(Phi[:, n:], U, spec)
                Ylr = (Ylr[..., :-1] / Ylr[..., -1:].clamp_min(1e-6)).view(b, self.h, l - n, self.p).permute(0, 2, 1, 3)
                lam = torch.sigmoid(self.alpha + self.lam_w(u.float()))[:, n:]           # (b, l-n, h), bounded in (0, 1)
                y = torch.cat([y[:, :n], y[:, n:] + (lam.unsqueeze(-1) * Ylr).to(y.dtype)], dim=1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = m.norm(y, z)
        return m.out_proj(y)


from zoology.mixers.mamba2 import Mamba2Block as _ZooMamba2Block


class SmatMamba2Block(_ZooMamba2Block):
    """Zoology's Mamba2Block (RMSNorm, no MLP, add->norm->mixer, Mamba init path) with the
    SMAT mixer from config.sequence_mixer in place of the vendored Mamba2.  Installed by
    zoo_smat_configs via `zoology.mixers.mamba2.Mamba2Block = SmatMamba2Block`, so
    block_type="Mamba2Block" gives SMAT arms exactly the reference arm's block and init."""
    def __init__(self, config, layer_idx=0, **kwargs):
        nn.Module.__init__(self)
        from zoology.mixers.mamba2 import RMSNorm
        self.residual_in_fp32, self.fused_add_norm = False, False
        self.norm = RMSNorm(config.d_model)
        self.mixer = config.sequence_mixer.instantiate(d_model=config.d_model, layer_idx=layer_idx)
        self.mlp = None


# ---------------------------------------------------------------------------
# Multi-resolution form, revision 3 ("MR3"): the Log-Linear-style fixes.
#   * heads x layers >= B (headdim knob) with a different mask offset per head,
#     so the union of the bank sees every profile;
#   * G's values are Mamba-2's write-gated stream dt_t * x_t (optionally with
#     Mamba-2's own decay, so profile states are slices of the SSD state);
#   * identity kernel (keys B_t, queries C_t, no feature map, no normaliser),
#     so G and the recurrence agree on what a match is;
#   * lambda_t = softplus(alpha[h, rho(t)] + w_h . u_t): per-head, per-row-type
#     bias plus an input-dependent term (the Log-Linear lambda projection).
# Parameter overhead over Mamba-2: lam_w (d_model x h) + alpha (h x B) only.
# Exactly Mamba-2 at init (alpha bias -> lambda ~ 3e-4).
# ---------------------------------------------------------------------------
import torch.nn.functional as F_
_EMB = {}                      # token embeddings of the current batch, stashed by the patched TokenEmbeddings (hash_src="embed")


def patch_token_embeddings():
    import zoology.model as zm
    if getattr(zm.TokenEmbeddings, "_smat_patched", False):
        return
    orig = zm.TokenEmbeddings.forward
    def fwd(self, *a, **k):
        out = orig(self, *a, **k); _EMB["x"] = out.detach(); return out
    zm.TokenEmbeddings.forward = fwd; zm.TokenEmbeddings._smat_patched = True


def delta_pool_profiles(Psi, Vb, beta, spec, eps=1e-6):
    """Option 1: per-profile DELTA-RULE memories instead of additive pools.

    Profile x holds the landmark tokens j with prof(j) = (j - n_P) mod N0, i.e. the stride-N0
    slice j = n_P + x + m*N0.  Each profile is processed as its own short sequence (length
    g = ceil(n_X / N0) ~ T^{1/d}, ~10 here) with the delta rule on L2-normalised keys:
        S <- S + beta_j * khat_j (v_j - khat_j^T S)^T          (S: r x p, read is phi^T S)
    so keys inside a profile are orthogonalised instead of superposed.  Same positional
    assignment and incidence as before, so the mask theory is untouched; only the state
    update rule changes.  Returns F: (nb, N0, r, p) like pool_profiles.
    """
    nb, L, r = Psi.shape
    p = Vb.shape[-1]
    n_P, N0, n_X = spec.n_P, spec.N0, spec.n_X
    g = -(-(n_X - n_P) // N0)
    m = torch.arange(g, device=Psi.device)
    idx = n_P + torch.arange(N0, device=Psi.device)[:, None] + m[None, :] * N0     # (N0, g)
    valid = (idx < n_X).float()                                                    # (N0, g)
    idx = idx.clamp(max=max(n_X - 1, 0))
    K = Psi[:, idx]                                                                # (nb, N0, g, r)
    V = Vb[:, idx]                                                                 # (nb, N0, g, p)
    Bt = beta[:, idx] * valid                                                      # (nb, N0, g), padded steps are no-ops
    K = K / (K.norm(dim=-1, keepdim=True) + eps)
    S = Psi.new_zeros(nb, N0, r, p)
    for t in range(g):
        k, v, b = K[:, :, t], V[:, :, t], Bt[:, :, t].unsqueeze(-1)                 # (nb,N0,r) (nb,N0,p) (nb,N0,1)
        pred = torch.einsum("bxr,bxrp->bxp", k, S)
        S = S + b.unsqueeze(-1) * k.unsqueeze(-1) * (v - pred).unsqueeze(-2)
    return S


def profile_keys(Psi, spec, eps=1e-6):
    """Summary key per profile: sum of the L2-normalised keys it holds, (nb, N0, r).
    A query whose key matches one stored in profile x has a large dot product with kappa_x."""
    n_P, N0, n_X = spec.n_P, spec.N0, spec.n_X
    g = -(-(n_X - n_P) // N0)
    idx = n_P + torch.arange(N0, device=Psi.device)[:, None] + torch.arange(g, device=Psi.device)[None, :] * N0
    valid = (idx < n_X).float().unsqueeze(-1)
    idx = idx.clamp(max=max(n_X - 1, 0))
    K = Psi[:, idx]
    K = K / (K.norm(dim=-1, keepdim=True) + eps)
    return (K * valid).sum(2)


class SmatMamba2MR(nn.Module):
    def __init__(self, d_model, layer_idx=0, d=2, d_state=128, headdim=None, mask_bank=True, n_P=0,
                 lam_bias=-8.0, detach_g=False, write_gate=True, g_decay=False, kernel="id", r=64, lam_act="softplus", pool="sum", read="type", reset=False, d_layers=None, g_floor=None,
                 hash_mode="point", hash_shift=1, hash_conv=False, hash_freeze=False, anneal_steps=0, balance_coef=0.01, hash_codim=None, hash_src="hidden", hash_freeze_after=0, hash_lr_scale=1.0, balance_gated=False, hash_conv_width=4, hash_ckpt=False, sparse_ops=False, g_bf16=False,
                 g_write_mode="shared", g_write_init=0.1, mamba_heads=None, **_):
        super().__init__()
        # read == "chash": the content-addressed assignment (content_addr.ContentAssign): profile = hash(key-side
        # source), row type = hash(query input), same C / pooling / decode as the paper.  hash_conv replaces the fixed
        # key-side shift by a learned depthwise causal conv (width 4) on u; anneal_steps anneals soft->hard read;
        # balance_coef * load-balance KL is returned through Zoology's get_auxiliary_loss hook.
        self.hash_mode, self.hash_shift, self.hash_conv, self.hash_freeze = hash_mode, hash_shift, hash_conv, hash_freeze
        self.hash_codim, self.hash_src = hash_codim, hash_src
        self.hash_freeze_after, self.hash_lr_scale, self.balance_gated = hash_freeze_after, hash_lr_scale, balance_gated
        self.anneal_steps, self.balance_coef, self._steps = anneal_steps, balance_coef, 0
        self.ca = nn.ModuleDict()
        self.d_model_ = d_model
        self.hash_conv_width, self.hash_ckpt = int(hash_conv_width), bool(hash_ckpt)   # hash_ckpt: recompute G in backward (activation checkpointing)
        self.sparse_ops, self.g_bf16 = bool(sparse_ops), bool(g_bf16)   # segmented pool/read (no N0 factor); run the G branch in bf16
        self.g_floor = g_floor                          # soft G: zeros of the incidence get weight g_floor ("q" -> 1/q) instead of 0
        if d_layers is not None:                       # per-layer d, e.g. [1, 2]: layer 0 plain Mamba-2, layer 1 SMAT d=2
            d = int(d_layers[layer_idx])
        self.lam_act, self.pool, self.read, self.reset = lam_act, pool, read, reset   # reset: run the recurrence separately on each half (paper mask: G is the only cross path)   # read: "type" (positional incidence) or "content" (softmax over ALL profiles by key match; option 2)   # pool: "sum" (additive profile states) or "delta" (per-profile delta rule)                              # "softplus" (unbounded) or "sigmoid" (lambda <= 1)
        d_inner = 2 * d_model
        headdim = headdim or min(64, d_inner)
        if mamba_heads is None:
            self.mixer = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2,
                                headdim=headdim, ngroups=1, chunk_size=64, use_mem_eff_path=False)
        else:
            # Explicit memory width independent of residual width. Initialize the
            # native SSD/conv/norm at H*P, then give it residual-width I/O maps.
            inner = int(mamba_heads) * int(headdim)
            self.mixer = Mamba2(d_model=inner, d_state=d_state, d_conv=4, expand=1,
                                headdim=headdim, ngroups=1, chunk_size=64, use_mem_eff_path=False)
            self.mixer.in_proj = nn.Linear(d_model, 2*inner + 2*d_state + int(mamba_heads), bias=False)
            self.mixer.out_proj = nn.Linear(inner, d_model, bias=False)
            self.mixer.d_model = d_model
            self.mixer.expand = inner / d_model
        m = self.mixer
        assert m.d_ssm == m.d_inner, "d_mlp path not supported"
        self.d, self.n_P = d, n_P
        self.detach_g, self.write_gate, self.g_decay, self.kernel, self.r = detach_g, write_gate, g_decay, kernel, r
        self.h, self.p, self.N = m.nheads, m.headdim, m.d_state
        # hash_src "ssm": hash the SSM output y_t (contextual: carries the recurrent state, e.g. counts) instead of the
        # layer input u_t; width h*p.  Keys and queries both hash y, so the alignment can use context in every layer.
        self.hash_dim = (self.h * self.p) if hash_src == "ssm" else d_model
        if read == "chash" and hash_conv:
            self.kconv = nn.Conv1d(self.hash_dim, self.hash_dim, self.hash_conv_width, groups=self.hash_dim, bias=False)
            nn.init.constant_(self.kconv.weight, 1.0 / self.hash_conv_width)   # uniform over lags; the model learns the lag(s)
        self.offsets = [layer_idx * self.h + i for i in range(self.h)] if mask_bank else None
        self.specs, self.phi = {}, None
        self.max_B = 65536                                  # alpha table width (row types; d=4 at T=16K has B~8400)
        if d >= 2:
            self.lam_w = nn.Linear(d_model, self.h)
            nn.init.zeros_(self.lam_w.weight); nn.init.zeros_(self.lam_w.bias)
            # the per-row-type alpha table is only used by the positional reads; with the content hash it would be 6M unused parameters
            self.alpha = nn.Parameter(torch.full((self.h, self.max_B if read != "chash" else 1), float(lam_bias)))
            if lam_act == "one":
                self.gscale = nn.Parameter(torch.ones(self.h))     # G added directly, learned per-head scale from 1
            if read == "content":
                self.tau = nn.Parameter(torch.zeros(self.h))          # per-head log-temperature of the profile softmax
            if pool == "delta":
                self.beta_w = nn.Linear(d_model, self.h)          # write strength beta_t = sigmoid(.) per head, init 0.5
                nn.init.zeros_(self.beta_w.weight); nn.init.zeros_(self.beta_w.bias)

        if g_write_mode not in ("shared", "independent", "residual"):
            raise ValueError(f"unknown G write mode: {g_write_mode}")
        if not 0 < g_write_init < 1:
            raise ValueError("g_write_init must be strictly between 0 and 1")
        if g_write_mode in ("independent", "residual") and not write_gate:
            raise ValueError("learned G write gate requires write_gate=True")
        self.g_write_mode = g_write_mode
        self.g_write_proj = None
        self.track_g_write_stats = False
        if d >= 2 and g_write_mode in ("independent", "residual"):
            # Preserve the RNG stream for the other layers, embeddings and lazy hash,
            # so shared/independent arms with the same seed start with identical common parameters.
            with torch.random.fork_rng(devices=[]):
                self.g_write_proj = nn.Linear(d_model, self.h)
            nn.init.zeros_(self.g_write_proj.weight)
            nn.init.constant_(self.g_write_proj.bias, 0.0 if g_write_mode == "residual"
                              else math.log(g_write_init / (1 - g_write_init)))

    def _g_write_weights(self, u, dts):
        """G writes: independent replaces dt; residual learns a multiplier around dt.

        This leaves the recurrence and G's optional decay untouched.
        """
        if self.g_write_proj is not None:
            gate = torch.sigmoid(self.g_write_proj(u.float()))
            return dts * (2 * gate) if self.g_write_mode == "residual" else gate
        return dts if self.write_gate else torch.ones_like(dts)

    def _spec(self, T, device):
        if T not in self.specs:
            chunk = max(16, T // 4)
            if self.read == "chash":
                # the hashed path needs no recent-block type cycle, so only the ambient-space bound
                # (q^{d-1} <= n) limits d: take the largest feasible d <= self.d
                import warnings
                spec, d_eff = None, self.d
                while d_eff >= 2 and spec is None:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            spec = build_mask(T, d_eff, chunk=chunk, n_P=self.n_P)
                    except ValueError:
                        d_eff -= 1
                self.specs[T] = to_device(spec, device) if spec is not None else None
            else:
                d_eff = min(self.d, d_max_geometric(T, chunk=chunk))
                self.specs[T] = to_device(build_mask(T, d_eff, chunk=chunk, n_P=self.n_P), device) if d_eff >= 2 else None
            if self.specs[T] is not None and self.read == "chash" and str(T) not in self.ca:
                from content_addr import ContentAssign
                dim = int(self.specs[T].dim)
                cods = self.hash_codim if isinstance(self.hash_codim, (list, tuple)) else [self.hash_codim] * self.h
                cods = [None if c is None else min(int(c), dim) for c in cods]
                groups = {}                                            # codim -> head indices
                for hi, c in enumerate(cods): groups.setdefault(c, []).append(hi)
                mods = nn.ModuleDict()
                for c, heads in groups.items():
                    mods[str(c)] = ContentAssign(len(heads), self.hash_dim, self.specs[T], mode=self.hash_mode, src="hidden" if self.hash_src == "ssm" else self.hash_src,
                                                 freeze=self.hash_freeze, shift=0 if self.hash_conv else self.hash_shift,
                                                 codim=c).to(device)
                    mods[str(c)].heads = heads
                    mods[str(c)].sparse_ops = self.sparse_ops
                    if self.hash_lr_scale != 1.0:
                        for prm in mods[str(c)].parameters():
                            if prm.requires_grad:
                                prm.register_hook(lambda g, sc=float(self.hash_lr_scale): g * sc)
                self.ca[str(T)] = mods
        return self.specs[T]

    def get_auxiliary_loss(self):
        if self.read != "chash" or not self.balance_coef:
            return 0.0
        cas = [c for mods in self.ca.values() for c in mods.values()]
        aux = [c.aux for c in cas if getattr(c, "aux", None) is not None]
        out = self.balance_coef * torch.stack(aux).sum() if aux else 0.0
        for c in cas: c.aux = None
        return out

    def forward(self, u, **_):
        m = self.mixer
        b, l, _ = u.shape
        spec = self._spec(l, u.device) if self.d >= 2 else None
        if self.read == "chash" and self.training and spec is not None:
            self._steps += 1
            if self.anneal_steps:
                for mods in self.ca.values():
                    for c in mods.values(): c.anneal = min(1.0, self._steps / float(self.anneal_steps))
            if self.hash_freeze_after and self._steps >= self.hash_freeze_after:
                for mods in self.ca.values():
                    for c in mods.values():
                        if not c.freeze:
                            for prm in c.parameters(): prm.requires_grad_(False)
                            c.freeze = True                                  # also drops the balance term
        zxbcdt = m.in_proj(u)
        A = -torch.exp(m.A_log.float())
        z, xBC, dt = torch.split(zxbcdt, [m.d_ssm, m.d_ssm + 2 * m.ngroups * m.d_state, m.nheads], dim=-1)
        fold = self.reset and spec is not None
        if fold:                                                   # paper mask: conv + recurrence restart at the boundary (n = l/2)
            assert 2 * spec.n == l
            xBC, dt = xBC.reshape(b * 2, l // 2, -1), dt.reshape(b * 2, l // 2, -1)
        xBC = causal_conv1d_fn(xBC.contiguous().transpose(1, 2), rearrange(m.conv1d.weight, "d 1 w -> d w"),
                               bias=m.conv1d.bias, activation=m.activation).transpose(1, 2)
        x, B, C = torch.split(xBC, [m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state], dim=-1)
        dt_limit_kwargs = {} if m.dt_limit == (0.0, float("inf")) else dict(dt_limit=m.dt_limit)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=m.headdim), dt, A,
            rearrange(B, "b l (g n) -> b l g n", g=m.ngroups), rearrange(C, "b l (g n) -> b l g n", g=m.ngroups),
            chunk_size=m.chunk_size, D=m.D, z=None, dt_bias=m.dt_bias, dt_softplus=True, **dt_limit_kwargs)  # (b, l, h, p)
        if fold:
            x, B, C, dt = (t.reshape(b, l, -1) for t in (x, B, C, dt))
            y = y.reshape(b, l, m.nheads, m.headdim)
        if spec is not None:
            n, Bn = spec.n, spec.B
            assert Bn <= self.max_B
            with torch.autocast(device_type=u.device.type, enabled=False):
                Bg, Cg, xg, dtg = (B, C, x, dt) if not self.detach_g else (B.detach(), C.detach(), x.detach(), dt.detach())
                Bf, Cf = Bg.float().reshape(b, l, self.N), Cg.float().reshape(b, l, self.N)
                if self.kernel == "id":
                    Phi_s, Psi_s = Cf, Bf                                                  # (b, l, N)
                else:
                    if self.phi is None:
                        self.phi = make_phi(self.N, self.r, device=u.device, dtype=torch.float32, seed=0)
                    Phi_s, Psi_s = self.phi(Cf), self.phi(Bf)
                rk = Phi_s.shape[-1]
                Phi = Phi_s.unsqueeze(1).expand(b, self.h, l, rk).reshape(b * self.h, l, rk)
                Psi = Psi_s.unsqueeze(1).expand(b, self.h, l, rk).reshape(b * self.h, l, rk)
                # Shared writes use dt; independent writes use a separate sigmoid.
                # Both retain the original optional exp(A * cumsum dt) decay.
                dts = F_.softplus(dtg.float() + m.dt_bias.float())                           # (b, l, h)
                if m.dt_limit != (0.0, float("inf")):
                    dts = dts.clamp(*m.dt_limit)
                xv = rearrange(xg.float(), "b l (h p) -> b l h p", p=self.p)
                write_weights = self._g_write_weights(u, dts)
                if self.track_g_write_stats:
                    self.last_g_write_weights = write_weights.detach()
                if self.write_gate:
                    xv = xv * write_weights.unsqueeze(-1)
                if self.g_decay:
                    la = torch.cumsum((A.view(1, -1, 1) * dts.permute(0, 2, 1)).contiguous(), dim=-1)   # (b, h, l), contiguous scan
                    key_w = torch.exp(la[:, :, n - 1:n] - la[:, :, :n])                      # (b, h, n): decay from key j to the boundary
                    qry_w = torch.exp(la[:, :, n:] - la[:, :, n - 1:n]).permute(0, 2, 1)     # (b, l-n, h): boundary -> query t
                    wfull = torch.ones(b, self.h, l, device=u.device, dtype=Psi.dtype)
                    wfull[:, :, :n] = key_w.to(Psi.dtype)
                    Psi = (Psi.view(b, self.h, l, rk) * wfull.unsqueeze(-1)).reshape(b * self.h, l, rk)   # one multiply, no cat/clone
                Vb = rearrange(xv, "b l h p -> (b h) l p")
                if self.kernel != "id":
                    Vb = torch.cat([Vb, torch.ones_like(Vb[..., :1])], dim=-1)
                if self.read == "chash":
                    U = None
                elif spec.n_X > 0:
                    if self.pool == "delta":
                        beta = torch.sigmoid(self.beta_w(u.float())).permute(0, 2, 1).reshape(b * self.h, l)   # (b h, l)
                        Fp = delta_pool_profiles(Psi, Vb, beta, spec)
                    else:
                        Fp = pool_profiles(Psi, Vb, spec, acc_dtype=torch.float32, backend="torch")
                    U = None if self.read == "content" else apply_incidence(Fp, spec, backend="torch")
                    if U is not None and self.g_floor:
                        # soft G: incident profiles get weight hi, the rest lo.  U[rho] = lo*F_total + (hi-lo)*U_C[rho]
                        if self.g_floor == "q":      hi, lo = 1.0, 1.0 / spec.q
                        elif self.g_floor == "pm":   hi, lo = 1.0 + 1.0 / spec.q, 1.0 - 1.0 / spec.q
                        elif self.g_floor == "m":    hi, lo = 1.0, 1.0 - 1.0 / spec.q
                        else:                        hi, lo = 1.0, float(self.g_floor)
                        U = lo * Fp.sum(1, keepdim=True) + (hi - lo) * U
                else:
                    Fp = torch.zeros(b * self.h, spec.N0, rk, Vb.shape[-1], device=u.device)
                    U = None if self.read == "content" else torch.zeros(b * self.h, Bn, rk, Vb.shape[-1], device=u.device)
                if self.read == "chash":
                    ksrc = None
                    hsrc = _EMB.get("x").float() if self.hash_src == "embed" else (y.reshape(b, l, -1).float() if self.hash_src == "ssm" else u.float())
                    if self.g_bf16:
                        Phi, Psi, Vb = Phi.to(torch.bfloat16), Psi.to(torch.bfloat16), Vb.to(torch.bfloat16)
                    if self.hash_conv:
                        write_src = hsrc.detach() if getattr(self, 'detach_write_hash_input', False) else hsrc
                        ksrc = self.kconv(F_.pad(write_src.transpose(1, 2), (self.hash_conv_width - 1, 0))).transpose(1, 2)   # causal, (b, l, C)
                    mods = self.ca[str(l)]
                    emb = _EMB.get("x").float() if self.hash_src == "embed" else None
                    kw_all = write_weights.permute(0, 2, 1).reshape(b * self.h, l)[:, :n] if self.balance_gated else None   # actual G write strength per landmark token
                    if len(mods) == 1:
                        camod = next(iter(mods.values()))
                        if self.hash_ckpt and torch.is_grad_enabled():
                            from torch.utils.checkpoint import checkpoint
                            Ylr = checkpoint(lambda U_, Ph_, Ps_, Vb_, E_, K_, W_: camod(U_, Ph_, Ps_, Vb_, n, emb=E_, key_src=K_, key_w=W_),
                                             hsrc, Phi, Psi, Vb, emb, ksrc, kw_all, use_reentrant=False)
                        else:
                            Ylr = camod(hsrc, Phi, Psi, Vb, n, emb=emb, key_src=ksrc, key_w=kw_all)   # (b h, l-n, p)
                    else:                                              # per-head codimension: run each head group's module
                        Ph, Ps, Vh = (t.view(b, self.h, l, -1) for t in (Phi, Psi, Vb))
                        Ylr = Vb.new_zeros(b, self.h, l - n, Vb.shape[-1])
                        for c in mods.values():
                            hs = c.heads
                            out = c(hsrc, Ph[:, hs].reshape(b * len(hs), l, -1), Ps[:, hs].reshape(b * len(hs), l, -1),
                                    Vh[:, hs].reshape(b * len(hs), l, -1), n, emb=emb, key_src=ksrc,
                                    key_w=None if kw_all is None else kw_all.view(b, self.h, n)[:, hs].reshape(b * len(hs), n))
                            Ylr[:, hs] = out.view(b, len(hs), l - n, -1)
                        Ylr = Ylr.reshape(b * self.h, l - n, -1)
                elif self.read == "content":
                    # option 2: content-addressed read over ALL profiles (incidence not applied) --
                    # score profile summary keys, softmax, weighted read of the profile states.
                    kap = profile_keys(Psi, spec)                                            # (b h, N0, r)
                    q = Phi[:, n:]                                                            # (b h, l-n, r)
                    temp = torch.exp(self.tau).repeat(b).view(b * self.h, 1, 1) / (rk ** 0.5)
                    w = torch.softmax(torch.einsum("btr,bxr->btx", q, kap) * temp, dim=-1)    # (b h, l-n, N0)
                    reads = torch.einsum("btr,bxrp->btxp", q, Fp)                             # (b h, l-n, N0, p)
                    Ylr = (w.unsqueeze(-1) * reads).sum(2)                                    # (b h, l-n, p)
                else:
                    if spec.n_P:
                        U = U + (Psi[:, :spec.n_P].transpose(-1, -2) @ Vb[:, :spec.n_P]).unsqueeze(1)
                    if self.offsets is not None:
                        Uh = U.view(b, self.h, Bn, rk, Vb.shape[-1])
                        Uh = torch.stack([torch.roll(Uh[:, i], shifts=-int(self.offsets[i]) % Bn, dims=1) for i in range(self.h)], dim=1)
                        U = Uh.reshape(b * self.h, Bn, rk, Vb.shape[-1])
                    Ylr = query_by_type(Phi[:, n:], U, spec)                                 # (b h, l-n, p[+1])
                if self.kernel != "id":
                    Ylr = Ylr[..., :-1] / Ylr[..., -1:].clamp_min(1e-6)
                Ylr = Ylr.view(b, self.h, l - n, self.p).permute(0, 2, 1, 3)                 # (b, l-n, h, p)
                if self.g_decay:
                    Ylr = Ylr * qry_w.unsqueeze(-1)
                rho = (torch.arange(l - n, device=u.device) % Bn)                            # row type of each recent query
                al = self.alpha if self.alpha.shape[1] > 1 else self.alpha.expand(self.h, Bn)     # content hash: one bias per head
                pre = al[:, rho].t().unsqueeze(0) + self.lam_w(u.float())[:, n:]                # (b, l-n, h)
                if self.lam_act == "one":
                    lam = self.gscale.view(1, 1, -1).expand(b, l - n, self.h)
                else:
                    lam = torch.sigmoid(pre) if self.lam_act == "sigmoid" else F_.softplus(pre)
                y = torch.cat([y[:, :n], y[:, n:] + (lam.unsqueeze(-1) * Ylr).to(y.dtype)], dim=1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = m.norm(y, z)
        return m.out_proj(y)


# ---------------------------------------------------------------------------
# Same method on a Gated DeltaNet base (fla.layers.gated_deltanet.GatedDeltaNet):
# GDN runs untouched over the whole sequence; SMAT adds N0 per-profile delta-rule
# memories built from GDN's own (L2-normalised) keys, values and beta, read by content
# (softmax over profile summary keys), lambda-gated onto the recent rows before GDN's
# gated RMSNorm and out-projection.  Exactly GDN at lambda = 0 (alpha init -8).
# expand_v = 1 so the per-head state is head_dim x head_dim (= Mamba-2's hd16/ds16).
# ---------------------------------------------------------------------------
from fla.layers.gated_deltanet import GatedDeltaNet as _FlaGDN
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _gdn_chunk


class SmatGDN(nn.Module):
    def __init__(self, d_model, layer_idx=0, d=2, headdim=16, expand_v=1, n_heads=None, lam_bias=-8.0,
                 lam_act="sigmoid", read="content", n_P=0, **_):
        super().__init__()
        n_heads = n_heads or max(1, (2 * d_model) // headdim)      # key_dim = 2 d_model, like Mamba-2's d_inner
        self.gdn = _FlaGDN(hidden_size=d_model, expand_v=expand_v, head_dim=headdim, num_heads=n_heads,
                           mode="chunk", use_gate=True, use_short_conv=True, layer_idx=layer_idx)
        g = self.gdn
        self.d, self.n_P, self.lam_act, self.read = d, n_P, lam_act, read
        self.h, self.r, self.p = g.num_heads, g.head_k_dim, g.head_v_dim
        self.specs, self.max_B = {}, 64
        if d >= 2:
            self.lam_w = nn.Linear(d_model, self.h)
            nn.init.zeros_(self.lam_w.weight); nn.init.zeros_(self.lam_w.bias)
            self.alpha = nn.Parameter(torch.full((self.h, self.max_B), float(lam_bias)))
            self.tau = nn.Parameter(torch.zeros(self.h))

    def _spec(self, T, device):
        if T not in self.specs:
            chunk = max(16, T // 4)
            d_eff = min(self.d, d_max_geometric(T, chunk=chunk))
            self.specs[T] = to_device(build_mask(T, d_eff, chunk=chunk, n_P=self.n_P), device) if d_eff >= 2 else None
        return self.specs[T]

    def forward(self, u, **_):
        g = self.gdn
        b, l, _ = u.shape
        spec = self._spec(l, u.device) if self.d >= 2 else None
        q, _ = g.q_conv1d(x=g.q_proj(u), cache=None, output_final_state=False, cu_seqlens=None)
        k, _ = g.k_conv1d(x=g.k_proj(u), cache=None, output_final_state=False, cu_seqlens=None)
        v, _ = g.v_conv1d(x=g.v_proj(u), cache=None, output_final_state=False, cu_seqlens=None)
        q, k = (rearrange(t, "... (h d) -> ... h d", d=g.head_k_dim) for t in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=g.head_v_dim)
        beta_raw = g.b_proj(u)
        o, _ = _gdn_chunk(q=q, k=k, v=v, g=g.a_proj(u), beta=beta_raw, A_log=g.A_log, dt_bias=g.dt_bias,
                          initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
                          use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                          allow_neg_eigval=g.allow_neg_eigval, state_v_first=True)      # (b, l, h, p)
        if spec is not None:
            n, Bn = spec.n, spec.B
            assert Bn <= self.max_B
            with torch.autocast(device_type=u.device.type, enabled=False):
                kf = rearrange(k.float(), "b l h r -> (b h) l r")
                qf = rearrange(q.float(), "b l h r -> (b h) l r")
                qf = qf / (qf.norm(dim=-1, keepdim=True) + 1e-6)                              # kernel also L2-normalises q
                vf = rearrange(v.float(), "b l h p -> (b h) l p")
                beta = torch.sigmoid(beta_raw.float()).permute(0, 2, 1).reshape(b * self.h, l)
                Fp = delta_pool_profiles(kf, vf, beta, spec)                                  # (b h, N0, r, p)
                kap = profile_keys(kf, spec)                                                  # (b h, N0, r)
                qq = qf[:, n:]
                temp = torch.exp(self.tau).repeat(b).view(b * self.h, 1, 1) / (self.r ** 0.5)
                w = torch.softmax(torch.einsum("btr,bxr->btx", qq, kap) * temp, dim=-1)
                reads = torch.einsum("btr,bxrp->btxp", qq, Fp)
                Ylr = (w.unsqueeze(-1) * reads).sum(2).view(b, self.h, l - n, self.p).permute(0, 2, 1, 3)
                rho = torch.arange(l - n, device=u.device) % Bn
                al = self.alpha if self.alpha.shape[1] > 1 else self.alpha.expand(self.h, Bn)     # content hash: one bias per head
                pre = al[:, rho].t().unsqueeze(0) + self.lam_w(u.float())[:, n:]
                lam = torch.sigmoid(pre) if self.lam_act == "sigmoid" else F_.softplus(pre)
                o = torch.cat([o[:, :n], o[:, n:] + (lam.unsqueeze(-1) * Ylr).to(o.dtype)], dim=1)
        gate = rearrange(g.g_proj(u), "... (h d) -> ... h d", d=g.head_v_dim)
        o = g.o_norm(o, gate)
        return g.o_proj(rearrange(o, "b t h d -> b t (h d)"))

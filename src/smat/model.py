"""Minimal byte-level language model on enwik8, with the attention op swapped
between causal softmax and SMAT (fixed mask, torch backends, trained end to end).

Everything but the attention op is identical across arms: byte embedding,
learned positional table, pre-norm blocks with a 4x GELU MLP, next-byte
cross-entropy.  SMAT uses ../smat unmodified: build_mask for the parameter-free
mask, make_phi / featurise for the elu+1 feature map, smat_attention for the op.

Reports validation bits per character (overall and by position bucket), the
per-byte perplexity, training-loss statistics, gradient health, step time and
peak memory.  Writes one CSV row per evaluation and a final JSON summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smat"))
from smat.mask import build_mask, d_max_geometric, materialise  # noqa: E402
from smat.attention import (featurise, make_phi, smat_attention, to_device, segmented_causal_scan,  # noqa: E402
                       pool_profiles, apply_incidence, query_by_type)
from smat.assign import ContentAssign                          # noqa: E402
import dataclasses                                                # noqa: E402

BUCKETS = [(0, 512), (512, 2048), (2048, 4096)]


def decayed_segmented_scan(Phi, Psi, Vb, logA, n, chunk=128):
    """Mamba/GLA-style selective decay in the causal scan:
        S_t = a_t S_{t-1} + psi_t vb_t^T,   y_t = Phi_t S_t,   a_t = exp(logA_t) in (0, 1]
    with one reset at ``n`` (the landmark/recent boundary), exactly like
    ``smat_attn.segmented_causal_scan``.  Chunked, fp32; within a chunk the
    weights exp(L_i - L_j), j <= i, and across chunks exp(L_i) are all <= 1.
    Shapes: Phi, Psi (nb, T, r), Vb (nb, T, p), logA (nb, T) -> (nb, T, p).
    """
    nb, T, r = Phi.shape
    p = Vb.shape[-1]
    C = min(chunk, T)
    pad = (-T) % C
    if pad:
        Phi = F.pad(Phi, (0, 0, 0, pad)); Psi = F.pad(Psi, (0, 0, 0, pad))
        Vb = F.pad(Vb, (0, 0, 0, pad)); logA = F.pad(logA, (0, pad))
    nc = (T + pad) // C
    nc0 = n // C
    Phi_c = Phi.float().reshape(nb, nc, C, r); Psi_c = Psi.float().reshape(nb, nc, C, r)
    Vb_c = Vb.float().reshape(nb, nc, C, p); la = logA.float().reshape(nb, nc, C)
    L = torch.cumsum(la, dim=-1)                                       # (nb, nc, C), <= 0
    tri = torch.ones(C, C, device=Phi.device).tril()
    # intra-chunk: A_ij = (phi_i . psi_j) exp(L_i - L_j), j <= i
    diff = (L.unsqueeze(-1) - L.unsqueeze(-2)).masked_fill(tri == 0, float("-inf"))
    W = torch.exp(diff)                                                # (nb, nc, C, C), mask before exp
    A = (Phi_c @ Psi_c.transpose(-1, -2)) * W
    intra = A @ Vb_c                                                   # (nb, nc, C, p)
    # chunk summaries with decay to the chunk end: D_b = sum_j exp(L_C - L_j) psi_j vb_j^T
    wj = torch.exp(L[..., -1:] - L)                                    # (nb, nc, C)
    D = (Psi_c * wj.unsqueeze(-1)).transpose(-1, -2) @ Vb_c            # (nb, nc, r, p)
    aC = torch.exp(L[..., -1])                                         # (nb, nc) total decay per chunk
    out = intra.clone()
    S = torch.zeros(nb, r, p, device=Phi.device, dtype=torch.float32)
    for b in range(nc):
        if b == nc0:
            S = torch.zeros_like(S)                                    # segment reset at n
        out[:, b] += (Phi_c[:, b] * torch.exp(L[:, b]).unsqueeze(-1)) @ S
        S = aC[:, b].view(nb, 1, 1) * S + D[:, b]
    return out.reshape(nb, T + pad, p)[:, :T]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
class Enwik8:
    """Standard 90M / 5M / 5M byte split.  Training windows are sampled at
    random offsets; validation is a fixed set of non-overlapping windows."""

    def __init__(self, path, T, n_val_windows, device, seed=0):
        raw = np.fromfile(path, dtype=np.uint8)
        self.train = torch.from_numpy(raw[:90_000_000].astype(np.int64))
        val = raw[90_000_000:95_000_000].astype(np.int64)
        n = min(n_val_windows, (len(val) - 1) // T)
        self.val = torch.from_numpy(np.stack(
            [val[i * T:i * T + T + 1] for i in range(n)]))          # (n, T+1)
        self.T, self.device = T, device
        self.gen = torch.Generator().manual_seed(seed)

    def batch(self, nb):
        ix = torch.randint(0, len(self.train) - self.T - 1, (nb,), generator=self.gen)
        x = torch.stack([self.train[i:i + self.T + 1] for i in ix])
        return x[:, :-1].to(self.device), x[:, 1:].to(self.device)

    def val_batches(self, nb):
        for i in range(0, len(self.val), nb):
            x = self.val[i:i + nb]
            yield x[:, :-1].to(self.device), x[:, 1:].to(self.device)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, arm, *, spec=None, r=64, learn_phi=False,
                 chunk=128, device=None, seed=0, qk_norm=False, triton_bwd=False,
                 no_lr=False, gate=False, phi_kind="elu1", taylor_dim=16, decay=False,
                 dir_head="linear", dir_soft=False, dir_window=4,
                 dt_init=(1e-3, 1e-1), A_init=(1.0, 16.0), type_offsets=None, scan_kernel="phi",
                 assign="positional", read_mode="point", pos_read="plane", hash_src="hidden",
                 hash_freeze=False, hash_shift=0, vocab=None, hash_conv=0):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.arm, self.spec, self.chunk = arm, spec, chunk
        self.phi_kind = phi_kind
        # mask bank: head h in this layer reads pooled state U_{(rho(t) + s_h) mod B} instead of
        # U_{rho(t)}: a different hyperplane per head/layer for the same row.  None = all zero.
        self.type_offsets = type_offsets
        # kernel used on the causal blocks (the scan) when decay is on:
        #   phi        positive features, ones column, one division with the landmark term (as before)
        #   phi-nonorm positive features, no normaliser
        #   signed     raw q, k (SSD-style), no normaliser -- Mamba-2's recurrence
        # The landmark block G always uses the positive phi kernel (the routing theory needs it).
        self.scan_kernel = scan_kernel
        # content-addressed assignment (Sec 3.2): prof/type become hashes of token
        # content instead of position.  None = the published positional maps.
        # positional assignment with point types: type(i) = i mod N0 reads one
        # profile state directly, so C is the identity and the hyperplane pass is
        # skipped.  This is the fourth cell of the (assignment x type family) grid
        # and the only one with no experiment behind it.
        # NB: opt-in via pos_read, never via read_mode, whose default is "point"
        # and which belongs to the content-addressed arm.
        self.pos_point = (arm == "smat" and assign == "positional" and pos_read == "point"
                          and spec is not None and getattr(spec, "kind", "") == "geometric"
                          and spec.d >= 2)
        self.spec_pt = dataclasses.replace(spec, B=spec.N0) if self.pos_point else None
        self.ca = None
        if (arm == "smat" and assign == "content" and spec is not None
                and getattr(spec, "kind", "") == "geometric" and spec.d >= 2):
            self.ca = ContentAssign(n_heads, d_model, spec, mode=read_mode,
                                    src=hash_src, freeze=hash_freeze, shift=hash_shift,
                                    vocab=vocab, conv_width=hash_conv,
                                    dir_head=dir_head, dir_soft=dir_soft,
                                    dir_window=dir_window)
        if scan_kernel == "signed":
            # SSD's B and C: learned linear maps of q, k up to the state dimension r
            # (d_state), signed, no nonlinearity.  State per head is r x d_head.
            self.P_q = nn.Linear(self.dh, r, bias=False)
            self.P_k = nn.Linear(self.dh, r, bias=False)
        # Mamba/GLA-style selective decay on the causal scan: per-head scalar
        # a_t = sigmoid(w.x_t + b)^(1/16), bias +4 -> a ~ 0.9989 at init.
        self.decay = decay                  # False | True | "coupled" | "mamba2"
        if decay == "mamba2":
            # Mamba-2 (SSD) scalar-per-head discretisation:
            #   dt_t = softplus(W x_t + b),  A = exp(A_log) > 0,
            #   a_t = exp(-dt_t A)  (decay),  write scaled by dt_t.
            # Init as in Mamba-2: dt in [0.001, 0.1] log-uniform via the bias,
            # A log-uniform in [1, 16].
            self.w_dt = nn.Linear(d_model, n_heads)
            self.A_log = nn.Parameter(torch.log(torch.empty(n_heads).uniform_(*A_init)))
            self.dt_init, self.A_init = dt_init, A_init
        elif decay:
            self.w_a = nn.Linear(d_model, n_heads)
        self.no_lr = no_lr          # control: the segmented scan only, no landmark block
        # gate 1: an input-dependent per-head scalar h_j on each landmark token's
        # key before pooling, so a token decides how strongly it is written into
        # its profile.  Only the long-range write is gated; the causal scan and
        # the mask are untouched.  Bias +1 starts the gates ~0.73 open.
        self.gate = gate
        if gate:
            self.w_h = nn.Linear(d_model, n_heads)
            nn.init.zeros_(self.w_h.weight); nn.init.constant_(self.w_h.bias, 1.0)
        self.gate_stats = None
        self.key_keep = None        # optional (nb, T) 0/1 gate on keys, e.g. an oracle noise mask
        self.probe = False          # stash (Phi, Psi) of the last forward for pollution probes
        self._last = None
        # "auto" selects the fused kernels, which are differentiable only once
        # smat_triton_bwd.patch() has been called (see --triton-bwd).
        self.backend = "auto" if triton_bwd else "torch"
        # QK-norm, applied identically in every arm: bounds the score inputs so
        # softmax logits (and the elu+1 features) cannot grow without limit.
        self.qn = nn.LayerNorm(self.dh) if qk_norm else nn.Identity()
        self.kn = nn.LayerNorm(self.dh) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if arm == "smat":
            if phi_kind == "taylor":
                # Based (Arora et al. 2024): phi(x) = [1, x, x (x) x / sqrt 2] on a
                # learned low-dim projection, so phi(q).phi(k) = 1 + s + s^2/2 with
                # s = q.k -- the 2nd-order Taylor expansion of exp(s), strictly
                # positive for every real s.  Feature count 1 + m + m^2.
                self.P_phi = nn.Linear(self.dh, taylor_dim, bias=False)
                self.phi = self._taylor
            elif learn_phi:
                self.W_phi = nn.Linear(self.dh, r, bias=False)
                self.phi = lambda x: F.elu(self.W_phi(x)) + 1.0
            else:
                self.phi = make_phi(self.dh, r, device=device, dtype=torch.float32, seed=seed)

    def _taylor(self, x):
        z = self.P_phi(x)                                            # (..., m)
        outer = (z.unsqueeze(-1) * z.unsqueeze(-2)).flatten(-2) / math.sqrt(2.0)
        return torch.cat([torch.ones_like(z[..., :1]), z, outer], dim=-1)

    def forward(self, x, emb=None, ids=None):
        nb, T, C = x.shape
        q, k, v = self.qkv(x).view(nb, T, 3, self.h, self.dh).unbind(2)
        q, k = self.qn(q), self.kn(k)
        if self.arm == "softmax":
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            o = o.transpose(1, 2)
        else:
            # heads folded into the batch; the op is single-head.  Run it in
            # fp32 outside autocast: it divides by a learned-feature normaliser.
            with torch.autocast(device_type=x.device.type, enabled=False):
                fold = lambda t: t.transpose(1, 2).reshape(nb * self.h, T, self.dh).float()
                keep = None
                if self.key_keep is not None:
                    keep = self.key_keep.float().unsqueeze(1).expand(nb, self.h, T).reshape(nb * self.h, T)
                Phi, Psi, Vb = featurise(fold(q), fold(k), fold(v), self.phi, h=keep)
                if self.probe:
                    self._last = (Phi.detach(), Psi.detach())
                if self.no_lr:
                    H = self._scan(Phi, Psi, Vb, x, nb, T)
                    o = H[..., :-1] / H[..., -1:].clamp_min(1e-6)   # guard: a fully-decayed row has ~0 mass
                elif self.decay and self.scan_kernel != "phi":
                    n = self.spec.n
                    if self.scan_kernel == "signed":
                        Sq, Sk, Sv = self.P_q(fold(q)), self.P_k(fold(k)), fold(v)
                    else:                                       # phi-nonorm
                        Sq, Sk, Sv = Phi, Psi, Vb[..., :-1]
                    o = self._scan(Sq, Sk, Sv, x, nb, T)                     # (nb*h, T, dv), unnormalised
                    if n < T:
                        if self.spec.n_X > 0:
                            Fp = pool_profiles(Psi, Vb, self.spec, acc_dtype=torch.float32, backend=self.backend)
                            U = apply_incidence(Fp, self.spec, backend=self.backend)
                        else:
                            U = torch.zeros(nb * self.h, self.spec.B, Phi.shape[-1], Vb.shape[-1],
                                            device=Phi.device, dtype=torch.float32)
                        if self.spec.n_P:
                            nP = self.spec.n_P
                            U = U + (Psi[:, :nP].transpose(-1, -2) @ Vb[:, :nP]).unsqueeze(1)
                        if self.type_offsets is not None:
                            B = self.spec.B
                            Uh = U.view(nb, self.h, B, U.shape[-2], U.shape[-1])
                            Uh = torch.stack([torch.roll(Uh[:, i], shifts=-int(self.type_offsets[i]) % B, dims=1)
                                              for i in range(self.h)], dim=1)
                            U = Uh.reshape(nb * self.h, B, U.shape[-2], U.shape[-1])
                        Ylr = query_by_type(Phi[:, n:], U, self.spec)                       # positive kernel, own normaliser
                        Ylr = Ylr[..., :-1] / Ylr[..., -1:].clamp_min(1e-6)
                        o = torch.cat([o[:, :n], o[:, n:] + Ylr], dim=1)
                elif self.decay:
                    n = self.spec.n
                    H = self._scan(Phi, Psi, Vb, x, nb, T)
                    if n < T and self.ca is not None:
                        Ylr = self.ca(x.float(), Phi, Psi, Vb, n,
                                      emb=None if emb is None else emb.float(), ids=ids)
                        H = torch.cat([H[:, :n], H[:, n:] + Ylr], dim=1)
                    elif n < T:
                        sp = self.spec_pt if self.pos_point else self.spec
                        if self.spec.n_X > 0:
                            Fp = pool_profiles(Psi, Vb, self.spec, acc_dtype=torch.float32, backend=self.backend)
                            U = Fp if self.pos_point else apply_incidence(Fp, self.spec, backend=self.backend)
                        else:                                   # every landmark is global: no profiled block
                            U = torch.zeros(nb * self.h, sp.B, Phi.shape[-1], Vb.shape[-1],
                                            device=Phi.device, dtype=torch.float32)
                        if self.spec.n_P:                       # Remark 3.4: the global channel, one shared state
                            nP = self.spec.n_P
                            b = Psi[:, :nP].transpose(-1, -2) @ Vb[:, :nP]
                            U = U + b.unsqueeze(1)
                        if self.type_offsets is not None:       # mask bank: roll each head's U by its offset
                            B = self.spec.B
                            Uh = U.view(nb, self.h, B, U.shape[-2], U.shape[-1])
                            Uh = torch.stack([torch.roll(Uh[:, i], shifts=-int(self.type_offsets[i]) % B, dims=1)
                                              for i in range(self.h)], dim=1)
                            U = Uh.reshape(nb * self.h, B, U.shape[-2], U.shape[-1])
                        Ylr = query_by_type(Phi[:, n:], U, sp)
                        H = torch.cat([H[:, :n], H[:, n:] + Ylr], dim=1)
                    o = H[..., :-1] / H[..., -1:].clamp_min(1e-6)   # guard: a fully-decayed row has ~0 mass
                elif self.gate:
                    n = self.spec.n
                    hg = torch.sigmoid(self.w_h(x).float())                       # (nb, T, H)
                    hf = hg.transpose(1, 2).reshape(nb * self.h, T, 1)
                    Fp = pool_profiles(Psi * hf, Vb, self.spec, acc_dtype=torch.float32,
                                       backend=self.backend)
                    U = apply_incidence(Fp, self.spec, backend=self.backend)
                    Ylr = query_by_type(Phi[:, n:], U, self.spec)
                    H = segmented_causal_scan(Phi, Psi, Vb, n=n, chunk=self.chunk,
                                              acc_dtype=torch.float32, backend=self.backend)
                    H = torch.cat([H[:, :n], H[:, n:] + Ylr], dim=1)
                    o = H[..., :-1] / H[..., -1:].clamp_min(1e-6)   # guard: a fully-decayed row has ~0 mass
                    with torch.no_grad():
                        g = hg[:, :n]
                        self.gate_stats = (g.mean().item(), (g < 0.1).float().mean().item(),
                                           (g > 0.9).float().mean().item())
                elif self.ca is not None:
                    # ungated control: the same content-addressed long-range branch
                    # on top of the plain segmented scan, so the gate can be ablated
                    # without changing anything else about the layer.
                    n = self.spec.n
                    H = segmented_causal_scan(Phi, Psi, Vb, n=n, chunk=self.chunk,
                                              acc_dtype=torch.float32, backend=self.backend)
                    if n < T:
                        Ylr = self.ca(x.float(), Phi, Psi, Vb, n,
                                      emb=None if emb is None else emb.float(), ids=ids)
                        H = torch.cat([H[:, :n], H[:, n:] + Ylr], dim=1)
                    o = H[..., :-1] / H[..., -1:].clamp_min(1e-6)
                else:
                    o = smat_attention(Phi, Psi, Vb, self.spec, chunk=self.chunk,
                                       acc_dtype=torch.float32,
                                       scan_backend=self.backend, incidence_backend=self.backend)
                o = o.view(nb, self.h, T, self.dh).transpose(1, 2).to(x.dtype)
        return self.out(o.reshape(nb, T, C))

    def _scan(self, Phi, Psi, Vb, x, nb, T):
        """the causal term: decayed if --decay, else the repo's segmented scan"""
        if self.decay == "mamba2":
            dt = F.softplus(self.w_dt(x).float())                        # (nb, T, H)
            logA = -(dt * torch.exp(self.A_log).float())                 # (nb, T, H)
            logA = logA.transpose(1, 2).reshape(nb * self.h, T)
            dtf = dt.transpose(1, 2).reshape(nb * self.h, T, 1)
            return decayed_segmented_scan(Phi, Psi * dtf, Vb, logA, n=self.spec.n, chunk=self.chunk)
        if self.decay:
            logA = (F.logsigmoid(self.w_a(x).float()) / 16.0)           # (nb, T, H)
            logA = logA.transpose(1, 2).reshape(nb * self.h, T)
            if self.decay == "coupled":
                # Mamba-style: the same gate scales the write, so a token with
                # a_t ~ 1 neither forgets nor writes (write scale 1 - a_t,
                # normalised by its init value so the scale starts at 1).
                w = (1.0 - torch.exp(logA)) / (1.0 - math.exp(F.logsigmoid(torch.tensor(4.0)).item() / 16.0))
                Psi = Psi * w.unsqueeze(-1)
            return decayed_segmented_scan(Phi, Psi, Vb, logA, n=self.spec.n, chunk=self.chunk)
        return segmented_causal_scan(Phi, Psi, Vb, n=self.spec.n, chunk=self.chunk,
                                     acc_dtype=torch.float32, backend=self.backend)


class Mamba2Mixer(nn.Module):
    """The real Mamba-2 mixer (mamba_ssm.Mamba2) as a drop-in for the attention op,
    for a reference arm inside the same block / harness.  d_state=128, expand=2,
    headdim = min(64, d_inner) as in the Mamba-2 defaults."""

    def __init__(self, d_model, chunk=64, **_):
        super().__init__()
        from mamba_ssm import Mamba2
        d_inner = 2 * d_model
        self.mixer = Mamba2(d_model=d_model, d_state=128, d_conv=4, expand=2,
                            headdim=min(64, d_inner), ngroups=1, chunk_size=min(chunk, 64))
        self.arm, self.gate_stats, self.probe, self._last, self.key_keep = "mamba2", None, False, None, None

    def forward(self, x):
        return self.mixer(x)


class Mamba2Smat(nn.Module):
    """SMAT layer with the real Mamba-2 mixer as the causal part.

    causal blocks:  Mamba2(x) with the state (and its conv) reset at the
                    landmark/recent boundary via seq_idx, so recent rows cannot
                    see landmarks through the recurrence -- only through G.
    long-range G:   SMAT's positive-kernel landmark block (pool by profile,
                    incidence, type-major readout, own normaliser; optional
                    global channel and per-head mask-bank offsets), projected
                    back to d_model and added to the recent rows.
    d=1: no reset, no G -> exactly Mamba-2.
    """

    def __init__(self, d_model, n_heads, spec=None, r=64, chunk=128, device=None, seed=0,
                 qk_norm=False, type_offsets=None, backend="torch", **_):
        super().__init__()
        from mamba_ssm import Mamba2
        d_inner = 2 * d_model
        self.mixer = Mamba2(d_model=d_model, d_state=128, d_conv=4, expand=2,
                            headdim=min(64, d_inner), ngroups=1, chunk_size=min(chunk, 64))
        self.arm, self.spec, self.chunk, self.backend = "mamba2smat", spec, chunk, backend
        self.h, self.dh = n_heads, d_model // n_heads
        self.type_offsets = type_offsets
        self.use_G = spec is not None and spec.d >= 2 and spec.n < spec.T
        self.gate_stats, self.probe, self._last, self.key_keep = None, False, None, None
        if self.use_G:
            self.qn = nn.LayerNorm(self.dh) if qk_norm else nn.Identity()
            self.kn = nn.LayerNorm(self.dh) if qk_norm else nn.Identity()
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.phi = make_phi(self.dh, r, device=device, dtype=torch.float32, seed=seed)

    def forward(self, x):
        nb, T, C = x.shape
        if not self.use_G:
            return self.mixer(x)
        n = self.spec.n
        seq_idx = torch.zeros(nb, T, dtype=torch.int32, device=x.device)
        seq_idx[:, n:] = 1                                   # reset at the boundary
        y = self.mixer(x, seq_idx=seq_idx)
        # ---- G: SMAT's landmark block, positive kernel ----
        q, k, v = self.qkv(x).view(nb, T, 3, self.h, self.dh).unbind(2)
        q, k = self.qn(q), self.kn(k)
        with torch.autocast(device_type=x.device.type, enabled=False):
            fold = lambda t: t.transpose(1, 2).reshape(nb * self.h, T, self.dh).float()
            Phi, Psi, Vb = featurise(fold(q), fold(k), fold(v), self.phi)
            if self.spec.n_X > 0:
                Fp = pool_profiles(Psi, Vb, self.spec, acc_dtype=torch.float32, backend=self.backend)
                U = apply_incidence(Fp, self.spec, backend=self.backend)
            else:
                U = torch.zeros(nb * self.h, self.spec.B, Phi.shape[-1], Vb.shape[-1],
                                device=x.device, dtype=torch.float32)
            if self.spec.n_P:
                nP = self.spec.n_P
                U = U + (Psi[:, :nP].transpose(-1, -2) @ Vb[:, :nP]).unsqueeze(1)
            if self.type_offsets is not None:
                B = self.spec.B
                Uh = U.view(nb, self.h, B, U.shape[-2], U.shape[-1])
                Uh = torch.stack([torch.roll(Uh[:, i], shifts=-int(self.type_offsets[i]) % B, dims=1)
                                  for i in range(self.h)], dim=1)
                U = Uh.reshape(nb * self.h, B, U.shape[-2], U.shape[-1])
            Ylr = query_by_type(Phi[:, n:], U, self.spec)
            Ylr = Ylr[..., :-1] / Ylr[..., -1:].clamp_min(1e-6)               # (nb*h, T_R, dv)
            Ylr = Ylr.view(nb, self.h, T - n, self.dh).transpose(1, 2).reshape(nb, T - n, C)
        y = torch.cat([y[:, :n], y[:, n:] + self.out(Ylr.to(y.dtype))], dim=1)
        return y


class Block(nn.Module):
    def __init__(self, d_model, n_heads, arm, short_conv=0, layer_idx=0, mask_bank=False, **kw):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        offsets = [layer_idx * n_heads + h for h in range(n_heads)] if mask_bank else None
        vocab = kw.pop("vocab", None)
        if arm == "mamba2":
            self.attn = Mamba2Mixer(d_model, chunk=kw.get("chunk", 64))
        elif arm == "mamba2smat":
            self.attn = Mamba2Smat(d_model, n_heads, type_offsets=offsets, **kw)
        else:
            self.attn = Attention(d_model, n_heads, arm, type_offsets=offsets,
                                  vocab=vocab, **kw)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))
        # optional causal depthwise short convolution before attention, the
        # local mixing that Zoology-style baselines carry; identical in every arm
        self.conv = (nn.Conv1d(d_model, d_model, short_conv, groups=d_model, padding=short_conv - 1)
                     if short_conv else None)
        self.ln0 = nn.LayerNorm(d_model) if short_conv else None

    def forward(self, x, emb=None, ids=None):
        if self.conv is not None:
            T = x.shape[1]
            x = x + self.conv(self.ln0(x).transpose(1, 2))[..., :T].transpose(1, 2)
        x = x + (self.attn(self.ln1(x), emb=emb, ids=ids) if isinstance(self.attn, Attention)
                 else self.attn(self.ln1(x)))
        return x + self.mlp(self.ln2(x))


class LM(nn.Module):
    def __init__(self, T, d_model, n_heads, n_layers, arm, vocab=256, short_conv=0, mask_bank=False, **kw):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(T, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, arm, short_conv=short_conv, layer_idx=i,
                                           mask_bank=mask_bank, vocab=vocab, **kw)
                                     for i in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init)
        for b in self.blocks:                      # the gate/decay init must survive the model-wide init
            if getattr(b.attn, "gate", False):
                nn.init.zeros_(b.attn.w_h.weight); nn.init.constant_(b.attn.w_h.bias, 1.0)
            if getattr(b.attn, "decay", False) == "mamba2":
                nn.init.zeros_(b.attn.w_dt.weight)
                lo, hi = b.attn.dt_init
                dt0 = torch.exp(torch.empty(b.attn.h).uniform_(math.log(lo), math.log(hi)))
                alo, ahi = b.attn.A_init          # log-uniform A, re-drawn here so the model-wide init cannot touch it
                with torch.no_grad():
                    b.attn.A_log.copy_(torch.empty(b.attn.h).uniform_(math.log(alo), math.log(ahi)))
                with torch.no_grad():
                    b.attn.w_dt.bias.copy_(dt0 + torch.log(-torch.expm1(-dt0)))   # inverse softplus
            elif getattr(b.attn, "decay", False):
                nn.init.zeros_(b.attn.w_a.weight); nn.init.constant_(b.attn.w_a.bias, 4.0)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        emb = self.tok(idx)
        x = emb + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, emb=emb, ids=idx)
        return self.head(self.ln_f(x))


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, nb, amp_dtype):
    model.eval()
    tot = torch.zeros(len(BUCKETS) + 1, dtype=torch.float64)
    cnt = torch.zeros(len(BUCKETS) + 1, dtype=torch.float64)
    for x, y in data.val_batches(nb):
        with torch.autocast(device_type=x.device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(x)
        nll = F.cross_entropy(logits.float().view(-1, 256), y.reshape(-1),
                              reduction="none").view_as(y).double().cpu()
        tot[0] += nll.sum(); cnt[0] += nll.numel()
        for i, (a, b) in enumerate(BUCKETS):
            seg = nll[:, a:min(b, nll.shape[1])]
            tot[i + 1] += seg.sum(); cnt[i + 1] += seg.numel()
    model.train()
    nats = (tot / cnt.clamp(min=1)).tolist()
    return {"val_bpc": nats[0] / math.log(2), "val_ppl": math.exp(nats[0]),
            **{f"val_bpc_{a}_{b}": nats[i + 1] / math.log(2)
               for i, (a, b) in enumerate(BUCKETS)}}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["softmax", "smat"], required=True)
    ap.add_argument("--d", type=int, default=1, help="SMAT mask dimension")
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--r", type=int, default=64, help="feature rank of phi")
    ap.add_argument("--learn-phi", action="store_true")
    ap.add_argument("--phi", choices=["elu1", "taylor"], default="elu1", help="feature map for SMAT arms")
    ap.add_argument("--decay", choices=["off", "on", "coupled", "mamba2"], default="off",
                    help="selective decay in the causal scan; 'coupled' also scales the write by 1 - a_t")
    ap.add_argument("--dt-min", type=float, default=1e-3)
    ap.add_argument("--dt-max", type=float, default=1e-1)
    ap.add_argument("--A-min", type=float, default=1.0)
    ap.add_argument("--A-max", type=float, default=16.0)
    ap.add_argument("--scan-kernel", choices=["phi", "phi-nonorm", "signed"], default="phi")
    ap.add_argument("--gate-wd", type=float, default=None,
                    help="weight decay on gate params (w_dt, A_log, w_a, w_h); default: same rule as other params")
    ap.add_argument("--taylor-dim", type=int, default=16)
    ap.add_argument("--qk-norm", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="gate 1: input-dependent key gate on the landmark write (long-range path only)")
    ap.add_argument("--no-lr", action="store_true",
                    help="control: keep the landmark/recent split but drop the long-range block")
    ap.add_argument("--triton-bwd", action="store_true",
                    help="train through the fused Triton kernels (smat_triton_bwd)")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--nb", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-windows", type=int, default=128)
    ap.add_argument("--eval-nb", type=int, default=8)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data", "enwik8"))
    ap.add_argument("--out", default="results/lm.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    global BUCKETS
    BUCKETS = [(0, 512), (512, args.T // 2), (args.T // 2, args.T)]     # position buckets scale with T
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    if args.triton_bwd:
        import smat_triton_bwd
        smat_triton_bwd.patch()
    amp = None if (args.no_amp or dev.type != "cuda") else torch.bfloat16
    name = ((args.arm if args.arm == "softmax" else f"smat_d{args.d}") + ("_qkn" if args.qk_norm else "")
            + ("_nolr" if args.no_lr else "") + ("_gate" if args.gate else "") + ("_taylor" if args.phi == "taylor" else "") + ("" if args.decay == "off" else f"_decay{args.decay if args.decay != 'on' else ''}")
            + ("_tri" if args.triton_bwd and args.arm == "smat" else ""))
    print(json.dumps({"run": name, **vars(args)}, indent=1, default=str), flush=True)

    spec, mask_stats = None, {"mask_nnz": args.T * (args.T + 1) // 2, "mask_density": (args.T + 1) / (2 * args.T)}
    if args.arm == "smat":
        dmax = d_max_geometric(args.T, chunk=args.chunk)
        if args.d > dmax:
            sys.exit(f"d={args.d} exceeds d_max({args.T})={dmax}")
        spec_np = build_mask(args.T, args.d, chunk=args.chunk)
        M = materialise(spec_np, dtype=np.uint8)
        mask_stats = {"mask_nnz": int(M.sum()), "mask_density": float(M.sum() / (args.T * args.T)),
                      "mask_n": spec_np.n, "mask_q": spec_np.q, "mask_B": spec_np.B}
        del M
        spec = to_device(spec_np, dev)
    print("mask:", mask_stats, flush=True)

    data = Enwik8(args.data, args.T, args.val_windows, dev, seed=args.seed)
    model = LM(args.T, args.d_model, args.heads, args.layers, args.arm, spec=spec,
               r=args.r, learn_phi=args.learn_phi, chunk=args.chunk, device=dev,
               seed=args.seed, qk_norm=args.qk_norm, triton_bwd=args.triton_bwd,
               no_lr=args.no_lr, gate=args.gate, phi_kind=args.phi, taylor_dim=args.taylor_dim,
               decay=(False if args.decay == "off" else (args.decay if args.decay != "on" else True)),
               dt_init=(args.dt_min, args.dt_max), A_init=(args.A_min, args.A_max), scan_kernel=args.scan_kernel).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", flush=True)

    is_gate = lambda n: any(k in n for k in ("w_dt", "A_log", "w_a", "w_h"))
    gate_p = [p for n, p in model.named_parameters() if is_gate(n)] if args.gate_wd is not None else []
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not (args.gate_wd is not None and is_gate(n))]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 and not (args.gate_wd is not None and is_gate(n))]
    groups = [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}]
    if gate_p:
        groups.append({"params": gate_p, "weight_decay": args.gate_wd})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
    sched = lambda s: (s + 1) / args.warmup if s < args.warmup else \
        0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, args.steps - args.warmup)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["run", "arm", "d", "step", "train_loss", "train_loss_med200", "val_bpc", "val_ppl",
              *[f"val_bpc_{a}_{b}" for a, b in BUCKETS], "grad_norm", "nonfinite_steps",
              "ms_per_step", "peak_MB", "params", "elapsed_s", "gate_mean", "gate_lo", "gate_hi",
              *mask_stats.keys()]
    new = not os.path.exists(args.out)
    fout = open(args.out, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
    if new:
        writer.writeheader()

    losses, nonfinite, t0 = [], 0, time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step_times = []
    row = None
    for step in range(1, args.steps + 1):
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step - 1)
        x, y = data.batch(args.nb)
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if not torch.isfinite(loss) or not torch.isfinite(gn):
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - ts)
        losses.append(loss.item())

        if step % 50 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  bpc {loss.item()/math.log(2):.3f}  "
                  f"gn {gn.item():.2f}  {1000*np.mean(step_times[-50:]):.0f} ms/step", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, data, args.eval_nb, amp)
            gs = [b.attn.gate_stats for b in model.blocks if b.attn.gate_stats is not None]
            gate = {k: (sum(g[i] for g in gs) / len(gs) if gs else "")
                    for i, k in enumerate(("gate_mean", "gate_lo", "gate_hi"))}
            row = {**gate, "run": name, "arm": args.arm, "d": args.d if args.arm == "smat" else "",
                   "step": step, "train_loss": losses[-1],
                   "train_loss_med200": float(np.median(losses[-200:])),
                   **ev, "grad_norm": gn.item(), "nonfinite_steps": nonfinite,
                   "ms_per_step": 1000 * float(np.mean(step_times[-args.eval_every:])),
                   "peak_MB": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0,
                   "params": n_params, "elapsed_s": time.time() - t0, **mask_stats}
            writer.writerow(row); fout.flush()
            print(f"== eval step {step}: val_bpc {ev['val_bpc']:.4f}  ppl {ev['val_ppl']:.3f}  "
                  + "  ".join(f"[{a}:{b}) {ev[f'val_bpc_{a}_{b}']:.3f}" for a, b in BUCKETS)
                  + f"  peak {row['peak_MB']:.0f} MB"
                  + (f"  gate mean {gate['gate_mean']:.3f} <0.1 {gate['gate_lo']:.2f} >0.9 {gate['gate_hi']:.2f}"
                     if gs else ""), flush=True)

    fout.close()
    with open(os.path.splitext(args.out)[0] + f"_{name}.json", "w") as f:
        json.dump(row, f, indent=1)
    print("done:", json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()

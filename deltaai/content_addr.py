"""Content-addressed assignment for the SMAT long-range block.

The published construction fixes ``prof(j) = (j - n_P) mod N0`` and
``type(i) = i mod B``: a token's group is decided by *where* it sits.  This
module replaces both maps with a hash of token content, keeping the finite
geometry (the points of F_q^{dim} and their hyperplanes) exactly as it is.
Only S and R in ``G = R C S^T`` change; C does not, so nnz(C) and every cost
and memory bound of the chunkwise theorem are untouched.

Two read modes, matching Sec. 3.2 of the draft:

  point  ``type(i) = {x(q_i)}``  -- the query reads its own cell.  At dim = 1
         (d = 2) every affine hyperplane of F_q^1 *is* a single point, so
         "point" and "plane" coincide there and the module collapses them.
  plane  ``type(i) = H_{a, a^T x(q_i)}`` -- the query reads the hyperplane of
         a hashed direction ``a`` that passes through its own cell.

Both contain ``x(q_i)``, so a query always sees every distant key whose hash
agrees with its own.

The hash reads the token's *layer input*, one projection for both roles.  This
is the hypothesis Prop. "Content-addressed recall" needs and the reason the
guarantee is about a token rather than about a position: the query role and the
key role of the same token hash identically by construction.  Hashing the
separate W_Q and W_K projections instead (as a first version of this module
did) breaks it -- a repeated key then lands in a different cell from the query
asking for it, and the visibility guarantee is vacuous.

Differentiability.  ``x(u)_l = floor(q sigma(W u)_l)`` has no gradient, so each
coordinate carries a two-point straight-through estimator: the forward value is
the hard bin (so every number reported is for the genuine hard binary mask) and
the backward pass sees a linear interpolation between the two bins adjacent to
``q sigma(W u)_l``.  The relaxation therefore touches 2^{dim} cells per token,
independent of q and of T; only the hard path survives into inference.  For
convenience at the sizes we run (N0 = 17 at d=2, 49 at d=3) the 2^{dim}-sparse
weights are materialised densely over the N0 cells, which is equivalent.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _grouped_planes(q: int, dim: int):
    """(dirs, grouped): normalised directions and, for each (direction, offset),
    the q^{dim-1} points of that affine hyperplane.  Same enumeration as
    ``smat_mask.build_mask`` (lex-ordered points, digit weight q^{dim-1-k}),
    recomputed here so this module does not depend on the type reordering that
    puts the planted hyperplanes first."""
    idx = np.arange(q ** dim, dtype=np.int64)
    pts = np.stack([(idx // q ** (dim - 1 - k)) % q for k in range(dim)], axis=1)
    nz = pts != 0
    lead = nz.argmax(axis=1)
    is_dir = nz.any(axis=1) & (pts[np.arange(q ** dim), lead] == 1)
    dirs = pts[is_dir]                                          # (D, dim)
    inner = (pts @ dirs.T) % q                                  # (N0, D)
    order = np.argsort(inner, axis=0, kind="stable")
    grouped = order.T.reshape(dirs.shape[0], q, q ** (dim - 1))  # (D, q, deg)
    return dirs, grouped


class ContentAssign(nn.Module):
    """Hashes tokens to profiles and queries to types, and runs the whole
    long-range branch (pool by profile, apply C, contract by type).

    Returns the *augmented* long-range term for the recent rows, shape
    ``(nb*h, T_R, p)``, to be added to the causal branch before the single
    division -- exactly where ``query_by_type`` sits in the positional path.
    """

    def __init__(self, n_heads, d_model, spec, mode="point", plant=True,
                 src="hidden", freeze=False, shift=0):
        super().__init__()
        assert spec.kind == "geometric" and spec.dim >= 1
        self.h = n_heads
        self.q, self.dim, self.N0 = int(spec.q), int(spec.dim), int(spec.N0)
        self.mode, self.plant = mode, plant
        # src   "hidden": hash the layer input.  "embed": hash the token embedding,
        #       which for a repeated token is identical by construction, so there is
        #       no invariance for the hash to learn.
        # shift the profile (key-side) source by this many positions: MQAR's payload
        #       sits one after its key, so the state that must be visible is the value
        #       position, whose profile must be the *preceding* key's hash.  Reading
        #       each position's own token puts the value in the wrong cell however
        #       good the hash is.  Queries are never shifted.
        # freeze: no gradient to the hash at all (a frozen random projection cannot
        #       collapse, so the balance penalty is unnecessary).
        self.src, self.freeze, self.shift = src, freeze, int(shift)
        if mode == "plane" and self.dim == 1:
            self.mode = "point"          # hyperplanes of F_q^1 are single points
        # one projection, one vector per token, both roles: this is what makes a
        # repeated key land in the cell its own query asks for.
        # sigma is the standard normal CDF applied to a unit-variance logit:
        # LayerNorm the token (per token, so the hash stays causal), project with
        # a unit-norm row, and Phi maps the resulting ~N(0,1) to ~U(0,1).  All q
        # bins are then equally occupied at initialisation, which is the spread
        # hypothesis Prop. "Content-addressed recall" needs; the balance penalty
        # below keeps it that way once W and gamma start moving.
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.W = nn.Parameter(torch.randn(n_heads, self.dim, d_model), requires_grad=not freeze)
        self.gamma = nn.Parameter(torch.ones(n_heads, self.dim), requires_grad=not freeze)
        self.b = nn.Parameter(torch.zeros(n_heads, self.dim), requires_grad=not freeze)
        if self.mode == "plane":
            dirs, grouped = _grouped_planes(self.q, self.dim)
            self.D = int(dirs.shape[0])
            self.Wd = nn.Parameter(torch.randn(n_heads, self.D, d_model) * d_model ** -0.5,
                                   requires_grad=not freeze)
            M = np.zeros((self.D, self.q, self.N0), dtype=np.float32)
            for e in range(self.D):
                for o in range(self.q):
                    M[e, o, grouped[e, o]] = 1.0
            self.register_buffer("M", torch.from_numpy(M))
        cols = spec.planted_cols if spec.planted_cols is not None else np.zeros(0, dtype=np.int64)
        self.register_buffer("plant_cols", torch.as_tensor(np.asarray(cols), dtype=torch.long))
        self.register_buffer("plant_cells",
                             torch.as_tensor([self.q ** (self.dim - l) for l in range(1, self.dim + 1)],
                                             dtype=torch.long))
        # anneal in [0,1] mixes the soft top-k read into the hard one: 0 is a genuine
        # soft read over the 2^{dim} cells adjacent to the query's logits, 1 is the
        # hard binary mask.  Scoring is done in the hash space, so the read contracts
        # only 2^{dim} states of size rp -- O(1) in T, unlike a soft read over all N0.
        self.anneal = 1.0
        self.last_soft_q = None   # soft cell distribution of the recent queries
        self.last_hard_k = None   # hard cell of each distant position
        self.probe = False
        self.last = None          # (key cells, query cells) when probing
        self.last_occ = None      # normalised entropy of the profile histogram
        self.last_dens = None     # mean row density of G over recent queries
        self.aux = None           # load-balancing penalty of the last forward

    # ------------------------------------------------------------------ hash
    def _cells(self, u, plant_at=False):
        """u: (nb, L, d_model) layer input -> (nb*h, L, N0) cell weights whose
        forward value is the one-hot of ``floor(q sigma(W u))`` and whose
        backward path is the two-bin interpolation."""
        nb, L, _ = u.shape
        Wn = self.W / self.W.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        z = torch.einsum("blm,hkm->bhlk", self.ln(u), Wn) * self.gamma[None, :, None, :] \
            + self.b[None, :, None, :]
        s = torch.special.ndtr(z)                                   # ~U(0,1) when z ~ N(0,1)
        s = (s * self.q).clamp(0.0, self.q - 1e-4)                  # (nb,h,L,dim) in [0,q)
        s = s.reshape(nb * self.h, L, self.dim)
        lo = s.floor()
        frac = s - lo
        lo = lo.long()
        hi = (lo + 1).clamp(max=self.q - 1)
        B = nb * self.h
        idx = torch.zeros(B, L, 1, dtype=torch.long, device=u.device)
        w = torch.ones(B, L, 1, device=s.device, dtype=s.dtype)
        ws = torch.ones(B, L, 1, device=s.device, dtype=s.dtype)
        a = float(self.anneal)
        for k in range(self.dim):
            place = self.q ** (self.dim - 1 - k)
            f = frac[..., k:k + 1]
            s_lo, s_hi = 1.0 - f, f                                  # the soft read
            e_lo = 1.0 + (s_lo - s_lo.detach())                      # forward 1, backward 1-frac
            e_hi = s_hi - s_hi.detach()                              # forward 0, backward frac
            w_lo, w_hi = (1 - a) * s_lo + a * e_lo, (1 - a) * s_hi + a * e_hi
            idx = torch.cat([idx + lo[..., k:k + 1] * place, idx + hi[..., k:k + 1] * place], dim=-1)
            w = torch.cat([w * w_lo, w * w_hi], dim=-1)
            ws = torch.cat([ws * s_lo, ws * s_hi], dim=-1)
        zer = torch.zeros(B, L, self.N0, device=s.device, dtype=s.dtype)
        out = zer.scatter_add(-1, idx, w)
        self._soft = zer.scatter_add(-1, idx, ws)                    # pure soft, for the agreement term
        # load balance: KL(mean bin occupancy || uniform), per head and coordinate.
        # Costs O(L) scatters plus O(q) on the mean, so no T*q term enters the cost.
        soft_lo, soft_hi = 1.0 - frac, frac
        occ = torch.zeros(B, self.dim, self.q, device=s.device, dtype=s.dtype)
        occ = occ.scatter_add(-1, lo.transpose(1, 2), soft_lo.transpose(1, 2))
        occ = occ.scatter_add(-1, hi.transpose(1, 2), soft_hi.transpose(1, 2))
        occ = occ / occ.sum(-1, keepdim=True).clamp_min(1e-6)
        self.aux = None if self.freeze else (occ * (occ.clamp_min(1e-9) * self.q).log()).sum(-1).mean()
        if plant_at and self.plant and self.plant_cols.numel():
            cols = self.plant_cols[self.plant_cols < L]
            if cols.numel():
                one = torch.zeros(cols.numel(), self.N0, device=out.device, dtype=out.dtype)
                one[torch.arange(cols.numel(), device=out.device), self.plant_cells[: cols.numel()]] = 1.0
                out = out.clone()
                out[:, cols] = one            # a fixed, input-independent profile
        return out

    # ------------------------------------------------------------- the branch
    def forward(self, u, Phi, Psi, Vb, n, emb=None):
        """u: (nb, T, d_model) layer input.  emb: (nb, T, d_model) token embeddings,
        used instead of u when src == "embed".  Phi, Psi: (nb*h, T, r).
        Vb: (nb*h, T, p).  n: landmark/recent boundary."""
        h = u if (self.src == "hidden" or emb is None) else emb
        hk = h
        if self.shift:                                              # prof(j) = hash(h_{j-shift})
            hk = torch.cat([h.new_zeros(h.shape[0], self.shift, h.shape[2]),
                            h[:, :-self.shift]], dim=1)
        Wk = self._cells(hk[:, :n], plant_at=True)                  # (b, n, N0)
        aux_k = self.aux
        self.last_hard_k = Wk.argmax(-1).detach()
        Wq = self._cells(h[:, n:])                                  # (b, T_R, N0)
        self.last_soft_q = self._soft
        self.aux = None if self.freeze else aux_k + self.aux
        if self.probe:
            kc, qc = Wk.argmax(-1).detach(), Wq.argmax(-1).detach()
            self.last = (kc, qc)
            # occupancy: normalised entropy of the distant profile histogram.
            # 1 = the hash spreads over all N0 cells, 0 = it has collapsed onto one.
            hist = torch.zeros(kc.shape[0], self.N0, device=kc.device)
            hist.scatter_add_(-1, kc, torch.ones_like(kc, dtype=hist.dtype))
            pr = hist / hist.sum(-1, keepdim=True).clamp_min(1.0)
            H = -(pr * pr.clamp_min(1e-9).log()).sum(-1)
            self.last_occ = (H / float(np.log(self.N0))).mean().item()
            # row density of G: the share of the distant block a recent query can
            # actually see.  1/N0 is a selective mask, 1.0 is no mask at all.  This
            # is the quantity the entropy above only proxies for: a hash may have
            # low entropy because it has collapsed, or because it has swept the
            # irrelevant tokens into their own cell, and only this separates them.
            self.last_dens = pr.gather(1, qc).mean().item()
        P, V = Psi[:, :n], Vb[:, :n]
        F = torch.stack([(P * Wk[:, :, x:x + 1]).transpose(-1, -2) @ V
                         for x in range(self.N0)], dim=1)           # (b, N0, r, p)
        Phi_r = Phi[:, n:]
        if self.mode == "point":
            Z = torch.einsum("bir,bxrp->bixp", Phi_r, F)            # (b, T_R, N0, p)
            return torch.einsum("bix,bixp->bip", Wq, Z)
        U = torch.einsum("eox,bxrp->beorp", self.M, F)              # (b, D, q, r, p)
        Z = torch.einsum("bir,beorp->bieop", Phi_r, U)              # (b, T_R, D, q, p)
        nb = u.shape[0]
        dl = torch.einsum("blm,hem->bhle", u[:, n:], self.Wd).reshape(Phi_r.shape[0], -1, self.D)
        pe = torch.softmax(dl, dim=-1)
        we = torch.zeros_like(pe).scatter_(-1, pe.argmax(-1, keepdim=True), 1.0) + (pe - pe.detach())
        po = torch.einsum("bix,eox->bieo", Wq, self.M)              # offset marginal per direction
        wo = torch.zeros_like(po).scatter_(-1, po.argmax(-1, keepdim=True), 1.0) + (po - po.detach())
        return torch.einsum("bie,bieo,bieop->bip", we, wo, Z)

    def agreement(self, pairs, valid):
        """The coupling term the quantization estimator does not provide: the
        query's cell distribution should put its mass on the cell where the state
        it needs actually sits.  ``pairs[b, :, 0]`` indexes recent queries,
        ``pairs[b, :, 1]`` the distant position each one must reach.  Gradient
        reaches every cell's logit, not just the two adjacent to a bin boundary."""
        if self.last_soft_q is None or self.last_hard_k is None:
            return None
        sq, hk = self.last_soft_q, self.last_hard_k                  # (B,m,N0), (B,n)
        nb = pairs.shape[0]
        rep = sq.shape[0] // nb
        i = pairs[..., 0].repeat_interleave(rep, 0).clamp_min(0)     # (B,P)
        j = pairs[..., 1].repeat_interleave(rep, 0).clamp_min(0)
        v = valid.repeat_interleave(rep, 0).to(sq.dtype)
        tgt = hk.gather(1, j)                                        # (B,P) cell to route to
        pq = sq.gather(1, i.unsqueeze(-1).expand(-1, -1, self.N0))   # (B,P,N0)
        lp = pq.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9).log()
        return -(lp * v).sum() / v.sum().clamp_min(1.0)

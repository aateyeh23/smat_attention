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


def _grouped_flats(q: int, dim: int, codim: int):
    """All distinct partitions of F_q^dim (arithmetic mod q) into the q^codim cosets of a
    codimension-``codim`` subspace: for each canonical ``codim x dim`` matrix A of full rank
    (distinct induced partitions only), coset of x = A x mod q.  Returns
    ``grouped``: (D_c, q^codim, q^(dim-codim)) point indices.  codim=1 reproduces the
    hyperplanes of ``_grouped_planes`` (same partitions, possibly different order)."""
    import itertools
    N0 = q ** dim
    idx = np.arange(N0, dtype=np.int64)
    pts = np.stack([(idx // q ** (dim - 1 - k)) % q for k in range(dim)], axis=1)   # (N0, dim)
    seen, grouped = set(), []
    deg = q ** (dim - codim)
    for rows in itertools.product(range(q), repeat=codim * dim):
        A = np.asarray(rows, dtype=np.int64).reshape(codim, dim)
        cid = (pts @ A.T) % q                                       # (N0, codim)
        key = (cid * (q ** np.arange(codim))[None, :]).sum(1)      # coset id per point
        _, first = np.unique(key, return_index=True)
        relabel = {key[i]: r for r, i in enumerate(sorted(first))}
        canon = tuple(relabel[k] for k in key)
        if len(relabel) != q ** codim or canon in seen:
            continue                                                # degenerate A, or same partition
        counts = np.bincount(np.asarray(canon), minlength=q ** codim)
        if (counts != deg).any():
            continue
        seen.add(canon)
        g = np.stack([np.flatnonzero(np.asarray(canon) == o) for o in range(q ** codim)])
        grouped.append(g)
    return np.stack(grouped)                                        # (D_c, q^codim, deg)


class ContentAssign(nn.Module):
    """Hashes tokens to profiles and queries to types, and runs the whole
    long-range branch (pool by profile, apply C, contract by type).

    Returns the *augmented* long-range term for the recent rows, shape
    ``(nb*h, T_R, p)``, to be added to the causal branch before the single
    division -- exactly where ``query_by_type`` sits in the positional path.
    """

    def __init__(self, n_heads, d_model, spec, mode="point", plant=True,
                 src="hidden", freeze=False, shift=0, codim=None):
        super().__init__()
        # codim: read from the codimension-c affine subspace through the query's own cell;
        # c=1 is the hyperplane ("plane"), c=dim is the single cell ("point").
        self.codim = None if codim is None else int(codim)
        if self.codim is not None:
            assert 1 <= self.codim <= int(spec.dim), (self.codim, spec.dim)
            mode = "point" if self.codim == int(spec.dim) else "plane"
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
            if self.codim is None or self.codim == 1:
                dirs, grouped = _grouped_planes(self.q, self.dim)
            else:
                grouped = _grouped_flats(self.q, self.dim, self.codim)
            self.D, self.n_cosets = int(grouped.shape[0]), int(grouped.shape[1])
            self.Wd = nn.Parameter(torch.randn(n_heads, self.D, d_model) * d_model ** -0.5,
                                   requires_grad=not freeze)
            M = np.zeros((self.D, self.n_cosets, self.N0), dtype=np.float32)
            for e in range(self.D):
                for o in range(self.n_cosets):
                    M[e, o, grouped[e, o]] = 1.0
            self.register_buffer("M", torch.from_numpy(M))
            self.register_buffer("coset_of", torch.from_numpy(M.argmax(1)))   # (D, N0): coset id of each cell per direction
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
        self.read_k = None
        self.read_backend = "torch"

    def enable_topk_reads(self, k, d_model, rank=None):
        """Learn a categorical score over types; gather exactly k distinct summaries.

        This reader is independent of the write hash. At dim=1, types are points.
        Selection is discrete; the selected logits receive ordinary softmax
        gradients. This does not make scoring/summary construction constant cost.
        """
        n_types = self.N0 if self.mode == "point" else self.D * self.n_cosets
        if not 1 <= k <= n_types:
            raise ValueError(f"read_k={k} exceeds {n_types} available types")
        self.read_k = int(k)
        if rank is not None and not 1 <= int(rank) <= d_model:
            raise ValueError('Read rank must be between1 and model width')
        read_dim = d_model if rank is None else int(rank)
        self.read_proj = None if rank is None else nn.Linear(d_model, read_dim, bias=False)
        if self.read_proj is not None:
            nn.init.normal_(self.read_proj.weight, std=d_model ** -0.5)
        self.W_read = nn.Parameter(torch.randn(self.h, n_types, read_dim) * read_dim ** -0.5)
        self.b_read = nn.Parameter(torch.zeros(self.h, n_types))
        if hasattr(self, "Wd"):
            self.Wd.requires_grad_(False)  # the top-k reader replaces direction selection

    def _topk_reads(self, u):
        if getattr(self, 'address_reads', False):
            from smat_address_reads import address_read_logits
            logits = address_read_logits(self, u)
            bias = torch.zeros_like(self.b_read)
        else:
            read_features = self.ln(u)
            if getattr(self, 'read_proj', None) is not None:
                read_features = self.read_proj(read_features)
            logits = torch.einsum("blm,hem->bhle", read_features, self.W_read)
            bias = self.b_read
        if self.training and getattr(self, 'read_noise_std', 0.) > 0:
            # Explore alternative discrete routes early, without extra reads.
            # The mixer anneals this noise to zero; evaluation is deterministic.
            logits = logits + torch.randn_like(logits) * self.read_noise_std
        if self.read_backend == "triton" and self.read_k == 4:
            from smat_read_triton import top4_softmax
            idx, weights = top4_softmax(logits, bias)
        else:
            logits = logits + bias[None, :, None, :]
            scores, idx = logits.flatten(0, 1).topk(self.read_k, dim=-1)
            weights = scores.softmax(-1)
        self.last_read_idx, self.last_read_weights = idx.detach(), weights.detach()
        return idx, weights

    # ------------------------------------------------------------------ hash
    def _cells(self, u, plant_at=False, tok_w=None, retain_neighbors=False):
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
        hi = ((lo + 1).remainder(self.q) if getattr(self, 'periodic_hash', False)
              else (lo + 1).clamp(max=self.q - 1))
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
        if retain_neighbors and getattr(self, 'symmetric_write_grad', False):
            from smat_address_reads import symmetric_write_routes
            idx, w, ws = symmetric_write_routes(s, self.q)
        if a >= 1.0 and getattr(self, "hard_k1", True) and not retain_neighbors:
            # hard mask: keep only the chosen cell (forward weight 1); its STE gradient still reaches W through
            # the chosen bin's interpolation weight.  4x less pooling / gather work than carrying all 2^dim pairs.
            idx1 = torch.zeros(B, L, 1, dtype=torch.long, device=u.device); w1 = torch.ones(B, L, 1, device=s.device, dtype=s.dtype)
            for k in range(self.dim):
                place = self.q ** (self.dim - 1 - k); f = frac[..., k:k + 1]
                idx1 = idx1 + lo[..., k:k + 1] * place; w1 = w1 * (1.0 + ((1.0 - f) - (1.0 - f).detach()))
            self._sparse = (idx1, w1)
        else:
            self._sparse = (idx, w)                                  # K = 2^dim (cell, weight) pairs per token
        if getattr(self, "sparse_ops", False):
            out = None                                              # dense (B, L, N0) never formed
            self._soft = None
        else:
            zer = torch.zeros(B, L, self.N0, device=s.device, dtype=s.dtype)
            out = zer.scatter_add(-1, idx, w)
            self._soft = zer.scatter_add(-1, idx, ws)                # pure soft, for the agreement term
        # load balance: KL(mean bin occupancy || uniform), per head and coordinate.
        # Costs O(L) scatters plus O(q) on the mean, so no T*q term enters the cost.
        soft_lo, soft_hi = 1.0 - frac, frac
        if tok_w is not None:                                        # gate-weighted occupancy: filler does not count
            tw = tok_w.reshape(B, L, 1).to(s.dtype)
            soft_lo, soft_hi = soft_lo * tw, soft_hi * tw
        occ = torch.zeros(B, self.dim, self.q, device=s.device, dtype=s.dtype)
        occ = occ.scatter_add(-1, lo.transpose(1, 2), soft_lo.transpose(1, 2))
        occ = occ.scatter_add(-1, hi.transpose(1, 2), soft_hi.transpose(1, 2))
        occ = occ / occ.sum(-1, keepdim=True).clamp_min(1e-6)
        self.aux = None if self.freeze else (occ * (occ.clamp_min(1e-9) * self.q).log()).sum(-1).mean()
        if not self.freeze and getattr(self, 'joint_hash_balance', False):
            from smat_address_reads import write_address_dependence
            joint_weights = ws if tok_w is None else ws * tok_w.reshape(B,L,1).to(ws.dtype)
            self.aux = self.aux + write_address_dependence(
                idx, joint_weights, self.h, self.q, self.dim)
        if plant_at and self.plant and self.plant_cols.numel():
            cols = self.plant_cols[self.plant_cols < L]
            if cols.numel():
                if out is None:                                     # sparse path: overwrite the (idx, w) pairs
                    idx2, w2 = (t.clone() for t in self._sparse)
                    idx2[:, cols, :] = self.plant_cells[: cols.numel()].view(1, -1, 1)
                    w2[:, cols, :] = 0.0; w2[:, cols, 0] = 1.0
                    self._sparse = (idx2, w2)
                else:
                    one = torch.zeros(cols.numel(), self.N0, device=out.device, dtype=out.dtype)
                    one[torch.arange(cols.numel(), device=out.device), self.plant_cells[: cols.numel()]] = 1.0
                    out = out.clone()
                    out[:, cols] = one            # a fixed, input-independent profile
        return out

    # ------------------------------------------------------------- the branch
    def forward(self, u, Phi, Psi, Vb, n, emb=None, key_src=None, key_w=None):
        """u: (nb, T, d_model) layer input.  emb: (nb, T, d_model) token embeddings,
        used instead of u when src == "embed".  Phi, Psi: (nb*h, T, r).
        Vb: (nb*h, T, p).  n: landmark/recent boundary.
        key_src: optional (nb, T, d_model) source for the KEY-side hash (e.g. a learned
        causal conv of u, replacing the fixed shift); queries always hash h."""
        h = u if (self.src == "hidden" or emb is None) else emb
        hk = h if key_src is None else key_src
        if self.shift:                                              # prof(j) = hash(h_{j-shift})
            hk = torch.cat([h.new_zeros(h.shape[0], self.shift, h.shape[2]),
                            h[:, :-self.shift]], dim=1)
        neighbor_grad = (getattr(self, "write_hash_neighbor_grad", False)
                         and torch.is_grad_enabled())
        if neighbor_grad and (self.read_k is None or not getattr(self, "sparse_ops", False)
                              or self.anneal != 1.0):
            raise ValueError("Neighbor write gradients require hard pooling and top-k reads")
        balance_w = (key_w.detach() if key_w is not None
                     and getattr(self, 'detach_balance_weights', False) else key_w)
        Wk = self._cells(hk[:, :n], plant_at=True, tok_w=balance_w,
                        retain_neighbors=neighbor_grad)          # None on the sparse path
        aux_k = self.aux
        sk = self._sparse
        if neighbor_grad:
            neighbor_idx, neighbor_weights = sk
            # At anneal=1 only the first route has a nonzero forward weight,
            # including planted anchors. Pool it once; train its hash separately.
            sk = (sk[0][..., :1], sk[1][..., :1].detach())
            self._sparse = sk
        self.last_hard_k = (Wk.argmax(-1) if Wk is not None else sk[0][..., 0]).detach()
        if self.read_k is not None:
            if not getattr(self, "sparse_ops", False):
                raise ValueError("Top-k summary reads require sparse_ops")
            from smat_pool_ops import pool_sorted, read_sorted
            P, V, Phi_r = Psi[:, :n], Vb[:, :n], Phi[:, n:]
            if getattr(self, "delta_updates", False):
                from smat_delta_pool import pool_delta
                memory = pool_delta(P, V, sk[0], sk[1].to(P.dtype), key_w, self.N0)
                if neighbor_grad and neighbor_weights.requires_grad:
                    from smat_write_hash_grad import with_delta_write_hash_gradient
                    memory = with_delta_write_hash_gradient(
                        memory, P, V, neighbor_idx, neighbor_weights, key_w, self.N0)
            else:
                memory = pool_sorted(P, V, sk[0], sk[1].to(P.dtype), self.N0)
                if neighbor_grad and neighbor_weights.requires_grad:
                    from smat_write_hash_grad import with_write_hash_gradient
                    memory = with_write_hash_gradient(
                        memory, P, V, neighbor_idx, neighbor_weights)
            if self.mode == "plane":
                # cuBLAS beats the sparse and custom dense kernels at training shapes.
                memory = torch.einsum("eox,bxrp->beorp", self.M.to(memory.dtype), memory)
                memory = memory.flatten(1, 2)
            idx, weights = self._topk_reads(h[:, n:])
            self.last_write_idx, self.last_write_weights = sk[0].detach(), sk[1].detach()
            self.aux = None if self.freeze else aux_k
            if self.read_backend == "triton":
                from smat_read_triton import read_topk
                return read_topk(Phi_r, idx, weights.to(Phi_r.dtype), memory)
            return read_sorted(Phi_r, idx, weights.to(Phi_r.dtype), memory)
        Wq = self._cells(h[:, n:])                                  # (b, T_R, N0)
        sq = self._sparse
        self.last_soft_q = self._soft
        self.aux = None if self.freeze else aux_k + self.aux
        if getattr(self, "sparse_ops", False):
            from smat_pool_ops import pool_sorted, read_sorted
            P, V, Phi_r = Psi[:, :n], Vb[:, :n], Phi[:, n:]
            dt = P.dtype
            if getattr(self, "delta_updates", False):
                from smat_delta_pool import pool_delta
                F = pool_delta(P, V, sk[0], sk[1].to(dt), key_w, self.N0)
            else:
                F = pool_sorted(P, V, sk[0], sk[1].to(dt), self.N0)
            if self.mode == "point":
                return read_sorted(Phi_r, sq[0], sq[1].to(dt), F)            # (b, T_R, p)    one bmm
            # affine-subspace read (codimension c; c=1 is the paper's hyperplane incidence): the query sums the
            # q^(dim-c) pooled cells of the coset through its own cell along a chosen direction.  Aggregate the
            # N0 cells to the D * q^c cosets once (an N0-sized index sum, no per-token work), then the query does
            # a point read into that table at (direction, coset of its cell).  The direction is chosen hard with a
            # one-sided straight-through factor on its softmax probability (gradient to the chosen direction's
            # logit only; the dense path scores all D directions).
            Bsz, r, p = F.shape[0], F.shape[2], F.shape[3]
            Fc = torch.einsum("eox,bxrp->beorp", self.M.to(F.dtype), F).reshape(Bsz, self.D * self.n_cosets, r, p)
            dl = torch.einsum("blm,hem->bhle", u[:, n:], self.Wd.to(u.dtype)).reshape(Phi_r.shape[0], -1, self.D)
            pe = torch.softmax(dl.float(), dim=-1)
            e_star = pe.argmax(-1, keepdim=True)                                                # (B, T_R, 1)
            pmax = pe.gather(-1, e_star)                                                        # (B, T_R, 1)
            ste = 1.0 + (pmax - pmax.detach())
            idx, w = sq
            coset = self.coset_of[e_star.expand_as(idx), idx]                                   # (B, T_R, K)
            y = read_sorted(Phi_r, e_star * self.n_cosets + coset, w.to(dt), Fc)
            return y * ste.to(y.dtype)
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
        Bsz, r, p = P.shape[0], P.shape[-1], V.shape[-1]
        # pool by cell as ONE batched matmul: F[x] = sum_j Wk[j,x] (Psi_j v_j^T)   -- (b, N0, r, p), no N0 loop
        PV = torch.einsum("bnr,bnp->bnrp", P, V).reshape(Bsz, n, r * p)
        F = torch.bmm(Wk.transpose(1, 2), PV).view(Bsz, self.N0, r, p)
        Phi_r = Phi[:, n:]
        if self.mode == "point":
            # the query's cell state U_i = sum_x Wq[i,x] F[x] as a bmm -- (b, T_R, r, p); never (b, T_R, N0, p)
            Uq = torch.bmm(Wq, F.reshape(Bsz, self.N0, r * p)).view(Bsz, -1, r, p)
            return torch.einsum("bir,birp->bip", Phi_r, Uq)
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

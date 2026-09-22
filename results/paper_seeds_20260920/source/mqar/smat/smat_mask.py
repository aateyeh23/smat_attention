#!/usr/bin/env python3
"""Construction and preprocessing of the revised SMAT masks ``M^(d)``.

This follows the revised Section 3.1.  With ``n`` landmark positions and
``T_R = T - n`` recent positions::

    M^(d) = [ N        0     ]        N = L_n   (geometric family)
            [ G      L_{T_R} ]        N = Z_k   (endpoint d = floor(log2 T))

The top-left block is the *causal triangle* ``L_n``, not the identity: every
landmark query sees its own prefix, so no position is isolated, ``d = 1`` is a
degenerate member of the family rather than a separate definition, and the
prefill schedule is a single chunked causal scan with one state reset.

Geometric family (2 <= d <= d_max)
----------------------------------
``prof: [n] -> F_q^{d-1}`` assigns a profile to each landmark column, ``H_0 ...
H_{B-1}`` are distinct affine hyperplanes of ``F_q^{d-1}``, ``rho(i) = i mod B``
is the type of recent query ``i``, and

    G_ij = 1{ prof(j) in H_{rho(i)} }.

With ``q = Theta(n^{1/d})`` and ``B = Theta(q^{d-1}) = Theta(T^{1-1/d})``.

The object whose sparsity governs cost is the *incidence* ``C in {0,1}^{B x N0}``
between hyperplanes and distinct profiles -- not ``G`` itself.  ``G = R C S^T``
where ``S`` pools landmark columns by profile and ``R`` is the one-hot row-type
map.  ``nnz(C) = B * q^{d-2} = Theta(T^{2-3/d})``, a factor ``q ~ T^{1/d}``
below the token-level incidence count ``I_tok`` that a direct scatter pays.

``d = 1`` is the degenerate member: ``B = 1``, ``H_0 = F_q^0`` is everything, so
``G`` is all-ones and ``M^(1) = L_T``.  It runs through exactly the same code
path -- no special case.

Endpoint (d = floor(log2 T))
----------------------------
``n = 2^k``, ``N = Z_k(a,x) = 1{x subseteq a}`` (the Boolean zeta mask, equal to
``L_2^{otimes k}``), and ``G_{ia} = 1{<u_i, a> = 0}`` over ``F_2`` with
``u_i = i mod 2^k``.  Here ``N`` is *not* replaced by ``L_n``: the zeta mask is
already a hierarchy and is what supplies the extra shattered bit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "MaskSpec",
    "build_mask",
    "incidence_audit",
    "materialise",
    "estimate_cost",
    "next_prime",
    "d_max_geometric",
    "planted_hyperplane_normals",
    "shatter_witness",
]


# ---------------------------------------------------------------------------
# small number theory
# ---------------------------------------------------------------------------

def _is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0:
        return False
    f = 3
    while f * f <= x:
        if x % f == 0:
            return False
        f += 2
    return True


def next_prime(x: int) -> int:
    """Smallest prime ``>= x`` (with ``x >= 2``).  Bertrand guarantees ``< 2x``."""
    p = max(2, int(x))
    while not _is_prime(p):
        p += 1
    return p


def _points(q: int, dim: int) -> np.ndarray:
    """The ``q^dim`` points of ``F_q^dim`` in lex order, shape ``(q^dim, dim)``.

    Index ``idx`` decodes as ``x_k = (idx // q^{dim-1-k}) mod q``, so the
    standard basis vector ``e_l`` (1-indexed) is point ``q^{dim-l}``.
    """
    if dim == 0:
        return np.zeros((1, 0), dtype=np.int64)
    idx = np.arange(q ** dim, dtype=np.int64)
    return np.stack([(idx // q ** (dim - 1 - k)) % q for k in range(dim)], axis=1)


def planted_hyperplane_normals(d: int) -> List[Tuple[np.ndarray, int]]:
    """The ``2^{d-1}`` planted pairs ``(a_R, b_R)``, indexed by ``R`` as a bitmask.

    For ``R`` a proper subset of ``[d-1]``: ``a_{R,i} = 0`` iff ``i in R``,
    ``b_R = 0``.  For ``R = [d-1]``: ``a_R = 1``, ``b_R = 1``.  These satisfy

        e_i in H_R  <=>  i in R,

    which is what makes all ``2^{d-1}`` subsets of the planted columns realisable
    (Eq. 3.5 of the draft).  Bit ``k`` of the mask means ``i = k+1 in R``.
    """
    dim = d - 1
    out: List[Tuple[np.ndarray, int]] = []
    full = (1 << dim) - 1
    for R in range(1 << dim):
        if R == full:
            a = np.ones(dim, dtype=np.int64)
            b = 1
        else:
            a = np.array([0 if (R >> k) & 1 else 1 for k in range(dim)],
                         dtype=np.int64)
            b = 0
        out.append((a, b))
    return out


def d_max_geometric(T: int, *, chunk: int = 1, n_P: int = 0) -> int:
    """Largest ``d`` for which the geometric family is defined at this ``T``.

    The binding constraints are the two of Eq. (3.6): the ambient space must fit
    in the landmark block (``q^{d-1} <= n - n_P``) and the recent block must
    complete two full type cycles (``T_R >= 2B``).  ``B >= 2^{d-1}`` is automatic
    because ``B >= q^{d-1} >= 2^{d-1}``.
    """
    d = 1
    while True:
        try:
            build_mask(T, d + 1, chunk=chunk, n_P=n_P, _probe=True)
        except (ValueError, NotImplementedError):
            return d
        d += 1


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------

@dataclass
class MaskSpec:
    """A precompiled ``M^(d)``.  Arrays are numpy; :func:`to_device` stages them."""

    T: int
    d: int
    kind: str                 # "geometric" | "endpoint"
    n: int                    # landmark positions [0, n)
    T_R: int                  # recent positions   [n, T)
    chunk_align: int = 1      # n is a multiple of this (see Sec 3.3)

    # --- geometric ---------------------------------------------------------
    q: int = 0                # field size
    dim: int = 0              # d - 1, the ambient dimension
    N0: int = 1               # |X_0|, the number of distinct profiles = q^{d-1}
    n_dirs: int = 0           # normalised directions  (q^{d-1}-1)/(q-1)
    B: int = 1                # number of hyperplanes / row types
    n_planted: int = 0        # 2^{d-1}, occupying type indices [0, n_planted)
    planted_cols: Optional[np.ndarray] = None   # (d-1,) landmark cols with prof=e_l

    # --- endpoint ----------------------------------------------------------
    k: int = 0                # n = 2^k

    # --- optional global channel (Remark 3.4) ------------------------------
    n_P: int = 0              # |P|; profiles are assigned on [n_P, n)

    # --- the incidence C in {0,1}^{B x N0}, stored CSR by row ---------------
    C_indptr: Optional[np.ndarray] = None       # (B+1,) int64
    C_indices: Optional[np.ndarray] = None      # (nnz,) int32, profile indices
    uniform_deg: int = 0                        # >0 iff every row of C has this degree

    build_times: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # torch views, filled by smat_attn.to_device
    C_indptr_t: object = None
    C_indices_t: object = None
    plane_pts_t: object = None                  # (B, uniform_deg) when uniform
    type_cols_t: object = None                  # (B, kappa) for the direct backend

    # -- derived -----------------------------------------------------------
    @property
    def n_X(self) -> int:
        """Number of profiled landmark columns."""
        return self.n - self.n_P

    @property
    def I_prof(self) -> int:
        """``nnz(C)`` -- the pooled long-range work, Theta(T^{2-3/d})."""
        return int(self.C_indices.size)

    @property
    def profile_degree(self) -> np.ndarray:
        """``nu(x)`` -- hyperplanes through each profile, shape ``(N0,)``."""
        return np.bincount(self.C_indices, minlength=self.N0).astype(np.int64)

    @property
    def I_tok(self) -> int:
        """Token-level incidences, ``sum_j nu(prof(j))`` -- Theta(T^{2-2/d}).

        This is what a direct per-column scatter pays, and it is the count the
        pooled schedule avoids.
        """
        return int(self.profile_degree[self.prof_of_column()].sum())

    def prof_of_column(self) -> np.ndarray:
        """Profile index of every profiled landmark column, shape ``(n_X,)``."""
        if self.kind == "endpoint":
            return np.arange(self.n_X, dtype=np.int64)
        return np.arange(self.n_X, dtype=np.int64) % self.N0

    def row_type(self, i):
        """Type ``rho(i)`` of recent row ``i`` (0-based within the recent block)."""
        return i % self.B

    def dense_C(self) -> np.ndarray:
        C = np.zeros((self.B, self.N0), dtype=np.int8)
        for h in range(self.B):
            C[h, self.C_indices[self.C_indptr[h]:self.C_indptr[h + 1]]] = 1
        return C

    def summary(self) -> str:
        if self.kind == "endpoint":
            return (f"M^({self.d}) endpoint T={self.T}: k={self.k} n={self.n} "
                    f"B={self.B} nnz(C)={self.I_prof:,}")
        return (f"M^({self.d}) T={self.T}: q={self.q} N0={self.N0} B={self.B} "
                f"deg={self.uniform_deg} n={self.n} T_R={self.T_R} "
                f"I_prof={self.I_prof:,} I_tok={self.I_tok:,}")


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def build_mask(T: int, d: int, *, chunk: int = 1, n_P: int = 0,
               B_frac: float = 1.0, max_I_prof: Optional[int] = None,
               _probe: bool = False) -> MaskSpec:
    """Build ``M^(d)`` for context length ``T``.

    ``chunk`` aligns the landmark/recent boundary to a multiple of the scan
    chunk width, so the segmented scan of Sec. 3.3 resets exactly on a chunk
    boundary (``n = c * floor(T / 2c)``).

    ``n_P`` enables the optional global channel of Remark 3.4: the first ``n_P``
    landmark columns are seen by *every* recent query and carry no profile.
    The default ``n_P = 0`` is the main construction.

    ``B_frac`` takes only a constant fraction of the available affine
    hyperplanes; the ``2^{d-1}`` planted ones are always retained, at type
    indices ``0 .. 2^{d-1}-1``.
    """
    if T < 8:
        raise ValueError(f"T must be >= 8, got {T}")
    d_ceiling = T.bit_length() - 1                    # floor(log2 T)
    if not (1 <= d <= d_ceiling):
        raise ValueError(f"need 1 <= d <= floor(log2 T) = {d_ceiling}, got d={d}")

    c = max(1, int(chunk))
    n = c * (T // (2 * c))
    if n <= 0:
        raise ValueError(f"chunk={c} too large for T={T}")
    T_R = T - n
    t0 = time.perf_counter()

    if d == d_ceiling and d >= 2:
        return _build_endpoint(T, d, c, t0, _probe=_probe)

    times: Dict[str, float] = {}
    warns: List[str] = []

    # ---- d = 1: the degenerate member.  B = 1, H_0 = everything, G all ones.
    if d == 1:
        spec = MaskSpec(
            T=T, d=1, kind="geometric", n=n, T_R=T_R, chunk_align=c,
            q=0, dim=0, N0=1, n_dirs=0, B=1, n_planted=0,
            planted_cols=np.zeros(0, dtype=np.int64), n_P=0,
            C_indptr=np.array([0, 1], dtype=np.int64),
            C_indices=np.zeros(1, dtype=np.int32),
            uniform_deg=1,
        )
        spec.build_times["total"] = time.perf_counter() - t0
        return spec

    # ---- field size and ambient space ------------------------------------
    t = time.perf_counter()
    dim = d - 1
    q = next_prime(int(np.ceil(n ** (1.0 / d))))
    N0 = q ** dim
    times["prime"] = time.perf_counter() - t

    if n_P < 0 or n_P >= n:
        raise ValueError(f"n_P={n_P} out of range for n={n}")
    if N0 > n - n_P:
        raise ValueError(
            f"T={T}, d={d}: ambient |F_q^{dim}| = {N0} exceeds the {n - n_P} "
            f"profiled landmark columns; T is too small for this d")

    # ---- points and normalised directions --------------------------------
    t = time.perf_counter()
    pts = _points(q, dim)                                     # (N0, dim)
    nz = pts != 0
    lead = nz.argmax(axis=1)
    is_dir = nz.any(axis=1) & (pts[np.arange(N0), lead] == 1)
    dirs = pts[is_dir]                                        # (D, dim)
    dir_index = -np.ones(N0, dtype=np.int64)
    dir_index[np.flatnonzero(is_dir)] = np.arange(dirs.shape[0])
    D = int(dirs.shape[0])
    assert D == (q ** dim - 1) // (q - 1), (D, q, dim)
    B_max = D * q
    times["directions"] = time.perf_counter() - t

    # ---- the two admissibility conditions of Eq. (3.6) --------------------
    n_planted = 1 << dim
    B = int(round(B_frac * B_max))
    B = max(n_planted, min(B, B_max))
    if B < n_planted:
        raise ValueError(f"T={T}, d={d}: B={B} < 2^(d-1)={n_planted}")
    if T_R < 2 * B:
        msg = (f"T={T}, d={d}: recent block T_R={T_R} < 2B={2 * B}; the type map "
               f"does not complete two cycles, so the VC lower bound does not apply")
        if _probe:
            raise ValueError(msg)
        warns.append(msg)

    deg = q ** (dim - 1)                      # points on each affine hyperplane
    I_prof = B * deg
    if max_I_prof is not None and I_prof > max_I_prof:
        # checked before the incidence is built: the build itself is
        # Theta(nnz(C)) in both time and memory, so a late check is no guard.
        raise MemoryError(f"T={T}, d={d}: nnz(C)={I_prof:,} exceeds "
                          f"max_I_prof={max_I_prof:,}")

    if _probe:
        return MaskSpec(T=T, d=d, kind="geometric", n=n, T_R=T_R, chunk_align=c,
                        q=q, dim=dim, N0=N0, n_dirs=D, B=B, n_planted=n_planted,
                        n_P=n_P, uniform_deg=deg)

    # ---- incidence: for every (direction, offset), the points on it -------
    # inner[:, e] = <a_e, x>.  A stable argsort of that one length-N0 column
    # cuts the space into the q parallel hyperplanes of direction e, each of
    # exactly q^{dim-1} points -- cheaper than sorting the nnz(C) pairs.
    t = time.perf_counter()
    inner = (pts @ dirs.T) % q                                # (N0, D)
    order = np.argsort(inner, axis=0, kind="stable")          # (N0, D)
    grouped = order.T.reshape(D, q, deg)                      # [e, b] -> points
    times["incidence"] = time.perf_counter() - t

    # ---- reorder so the planted hyperplanes occupy types 0 .. 2^{d-1}-1 ---
    t = time.perf_counter()
    planted_h = []
    for a, b in planted_hyperplane_normals(d):
        a_idx = int((a * (q ** np.arange(dim - 1, -1, -1))).sum())
        e = int(dir_index[a_idx])
        assert e >= 0, f"planted normal {a} is not a normalised direction"
        planted_h.append(e * q + b)
    assert len(set(planted_h)) == n_planted, "planted hyperplanes are not distinct"

    rest = np.setdiff1d(np.arange(B_max, dtype=np.int64),
                        np.array(planted_h, dtype=np.int64), assume_unique=False)
    h_order = np.concatenate([np.array(planted_h, dtype=np.int64), rest])[:B]

    flat = grouped.reshape(B_max, deg)
    plane_pts = flat[h_order].astype(np.int32)                # (B, deg)
    times["reorder"] = time.perf_counter() - t

    # ---- planted columns: prof(j) = e_l  =>  point index q^{dim-l} --------
    planted_cols = np.array([n_P + q ** (dim - l) for l in range(1, dim + 1)],
                            dtype=np.int64)
    assert planted_cols.max(initial=0) < n

    times["total"] = time.perf_counter() - t0
    return MaskSpec(
        T=T, d=d, kind="geometric", n=n, T_R=T_R, chunk_align=c,
        q=q, dim=dim, N0=N0, n_dirs=D, B=B, n_planted=n_planted,
        planted_cols=planted_cols, n_P=n_P,
        C_indptr=(np.arange(B + 1, dtype=np.int64) * deg),
        C_indices=plane_pts.reshape(-1),
        uniform_deg=deg,
        build_times=times, warnings=warns,
    )


def _build_endpoint(T: int, d: int, c: int, t0: float, *,
                    _probe: bool = False) -> MaskSpec:
    """``d = floor(log2 T)``: Boolean zeta landmarks, complemented Walsh rows."""
    k = d - 1
    n = 1 << k
    T_R = T - n
    B = 1 << k
    warns: List[str] = []
    if T_R < B:
        warns.append(
            f"T={T}: recent block T_R={T_R} < B={B}; fewer than one full Walsh "
            f"cycle, so not every row type occurs and the lower bound fails.")
    if _probe:
        raise NotImplementedError("endpoint is not part of the geometric family")

    # C[h, a] = 1{<u_h, a> = 0} over F_2, u_h = h.  Row 0 is all of F_2^k.
    a = np.arange(1 << k, dtype=np.int64)
    par = np.zeros((B, 1 << k), dtype=bool)
    for h in range(B):
        par[h] = (np.bitwise_count(np.bitwise_and(a, h)) % 2) == 0
    indptr = np.concatenate([[0], np.cumsum(par.sum(1))]).astype(np.int64)
    indices = np.tile(a, (B, 1))[par].astype(np.int32)

    spec = MaskSpec(
        T=T, d=d, kind="endpoint", n=n, T_R=T_R, chunk_align=c,
        q=2, dim=k, N0=n, n_dirs=B, B=B, n_planted=0,
        planted_cols=np.array([1 << j for j in range(k)], dtype=np.int64),
        k=k, n_P=0,
        C_indptr=indptr, C_indices=indices,
        uniform_deg=0,                       # ragged: row 0 has degree 2^k
        warnings=warns,
    )
    spec.build_times["total"] = time.perf_counter() - t0
    return spec


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def incidence_audit(spec: MaskSpec) -> Dict[str, object]:
    """Verify the counting identities the cost bounds rest on.

    ``nnz(C) = sum_x nu(x)``, every affine hyperplane of ``F_q^{d-1}`` carries
    ``q^{d-2}`` points, every point lies on ``(q^{d-1}-1)/(q-1)`` of them (when
    the full family is used), and ``I_tok = sum_j nu(prof(j))`` is a factor
    ``~ n / N0 ~ q`` larger than ``nnz(C)``.
    """
    nu = spec.profile_degree
    row_deg = np.diff(spec.C_indptr)
    out: Dict[str, object] = {
        "kind": spec.kind, "d": spec.d, "T": spec.T,
        "q": spec.q, "N0": spec.N0, "B": spec.B,
        "I_prof": spec.I_prof, "I_tok": spec.I_tok,
        "pool_gain": spec.I_tok / max(1, spec.I_prof),
        "row_deg_min": int(row_deg.min()), "row_deg_max": int(row_deg.max()),
        "col_deg_min": int(nu.min()), "col_deg_max": int(nu.max()),
        "nnz_eq_sum_nu": bool(spec.I_prof == int(nu.sum())),
        "rows_distinct_cols": bool(all(
            np.unique(spec.C_indices[spec.C_indptr[h]:spec.C_indptr[h + 1]]).size
            == row_deg[h] for h in range(0, spec.B, max(1, spec.B // 32)))),
    }

    if spec.kind == "geometric" and spec.d >= 2:
        q, dim = spec.q, spec.dim
        out["deg_is_q_pow_d_minus_2"] = bool(row_deg.min() == row_deg.max()
                                             == q ** (dim - 1))
        full = spec.B == spec.n_dirs * q
        out["full_hyperplane_family"] = bool(full)
        if full:
            out["nu_is_D"] = bool(nu.min() == nu.max() == spec.n_dirs)
        # Eq. (3.5): e_i in H_R  <=>  i in R, over the planted block
        C = spec.dense_C()
        pcols = spec.planted_cols - spec.n_P            # profile index of e_l
        ok = True
        for R in range(spec.n_planted):
            for l in range(dim):
                want = bool((R >> l) & 1)
                got = bool(C[R, pcols[l]])
                ok &= (want == got)
        out["planted_identity"] = bool(ok)
        out["B_ge_2_pow_d_minus_1"] = bool(spec.B >= (1 << dim))
        out["T_R_ge_2B"] = bool(spec.T_R >= 2 * spec.B)

    if spec.kind == "endpoint":
        out["T_R_ge_B"] = bool(spec.T_R >= spec.B)

    out["exact"] = bool(out["nnz_eq_sum_nu"] and out["rows_distinct_cols"]
                        and out.get("planted_identity", True)
                        and out.get("deg_is_q_pow_d_minus_2", True))
    return out


# ---------------------------------------------------------------------------
# the lower-bound witness
# ---------------------------------------------------------------------------

def shatter_witness(spec: MaskSpec):
    """The ``d`` columns the construction shatters, checked from the definition.

    Deciding VC exactly is LOGNP-complete, so the exact search in
    ``vc_dimension.py`` only reaches small ``T``.  The *lower* bound, though,
    is constructive and can be replayed at any ``T`` in ``O(2^d d)`` without
    materialising the mask.

    Returns ``(cols, ok, rows)``: the shattered columns, whether all ``2^|cols|``
    patterns were realised, and the witnessing row for each pattern.
    """
    C = spec.dense_C()

    if spec.kind == "endpoint":
        # The draft takes the recent column at t = B+1 and so asks for m >= 2B,
        # which fails at an exact power of two where T_R = B exactly.  The
        # *first* recent column works instead and needs only T_R >= B: every
        # recent row contains it (i >= 0 always) and no zeta row does, so the
        # 2^k zeta rows supply every pattern with a 0 there and the B recent
        # rows -- one per Walsh type -- supply every pattern with a 1.
        k, n = spec.k, spec.n
        cols = [1 << j for j in range(k)] + [n]
        ok, rows = True, {}
        for pat in range(1 << (k + 1)):
            want = [bool((pat >> b) & 1) for b in range(k + 1)]
            if not want[-1]:
                a = sum(cols[b] for b in range(k) if want[b])
                got = [((c & a) == c) if c < n else False for c in cols]
                found = ("zeta", a) if got == want else None
            else:
                found = None
                for i in range(0, min(spec.T_R, spec.B)):
                    got = [bool(C[i % spec.B, c]) if c < n else (i >= c - n)
                           for c in cols]
                    if got == want:
                        found = ("recent", i)
                        break
            ok &= found is not None
            rows[pat] = found
        return cols, bool(ok), rows

    dim, n = spec.dim, spec.n
    t = spec.B                                  # local index of the recent column
    cols = list(spec.planted_cols) + [n + t]
    prof = spec.prof_of_column()

    def realise(i, j):
        """Entry ``M[n+i, j]`` from the definition, without building M."""
        if j < spec.n_P:
            return True
        if j < n:
            return bool(C[i % spec.B, prof[j - spec.n_P]])
        return i >= (j - n)

    ok, rows = True, {}
    for R in range(1 << dim):
        for bit in (0, 1):
            i = R + (spec.B if bit else 0)       # type R, before or after t
            want = [bool((R >> l) & 1) for l in range(dim)] + [bool(bit)]
            ok &= (i < spec.T_R) and ([realise(i, c) for c in cols] == want)
            rows[(R, bit)] = i
    return cols, bool(ok), rows


# ---------------------------------------------------------------------------
# dense materialisation (verification only -- O(T^2))
# ---------------------------------------------------------------------------

def materialise(spec: MaskSpec, dtype=np.float64) -> np.ndarray:
    """The dense ``T x T`` 0/1 mask.  For verification at small ``T`` only."""
    T, n, T_R = spec.T, spec.n, spec.T_R
    M = np.zeros((T, T), dtype=dtype)

    if spec.kind == "endpoint":
        a = np.arange(n, dtype=np.int64)
        # Z_k(a, x) = 1{x subseteq a}
        M[:n, :n] = ((a[:, None] & a[None, :]) == a[None, :]).astype(dtype)
    else:
        M[:n, :n] = np.tril(np.ones((n, n), dtype=dtype))          # L_n

    M[n:, n:] = np.tril(np.ones((T_R, T_R), dtype=dtype))          # L_{T_R}
    if spec.n_P:
        M[n:, :spec.n_P] = 1.0                                     # optional P

    C = spec.dense_C()
    rho = np.arange(T_R, dtype=np.int64) % spec.B
    prof = spec.prof_of_column()
    M[n:, spec.n_P:n] = C[np.ix_(rho, prof)].astype(dtype)         # G = R C S^T
    return M


# ---------------------------------------------------------------------------
# cost prediction (no allocation)
# ---------------------------------------------------------------------------

def estimate_cost(T: int, d: int, r: int, d_v: int, *, chunk: int = 128,
                  n_P: int = 0) -> Dict[str, float]:
    """Predict ``nnz(C)``, MACs and bytes for a config without building it."""
    try:
        s = build_mask(T, d, chunk=chunk, n_P=n_P, _probe=True)
    except NotImplementedError:
        s = None
    except ValueError as e:
        return {"feasible": False, "reason": str(e)}

    p = d_v + 1
    if s is None or s.kind == "endpoint":
        k = d - 1
        B = N0 = 1 << k
        I_prof = B * (1 << max(0, k - 1))
        q = 2
    else:
        B, N0, q = s.B, s.N0, s.q
        I_prof = B * (s.uniform_deg if d >= 2 else 1)

    macs = (2 * T * r * d_v                    # feature maps
            + T * chunk * (r + p)              # intra-chunk scores and contraction
            + 3 * T * r * p                    # pooling, chunk summaries, states
            + I_prof * r * p)                  # the incidence contraction U = C F
    return {
        "feasible": True, "q": q, "B": B, "N0": N0, "I_prof": I_prof,
        "macs": float(macs),
        "table_bytes_fp32": 4.0 * (B + N0) * r * p,
        "dense_mask_bytes_fp32": 4.0 * T * T,
        "flash_macs": float(T) * T * (r + d_v),
    }


if __name__ == "__main__":
    for T in (4096, 16384, 65536, 262144):
        for d in (1, 2, 3, 4):
            try:
                s = build_mask(T, d, chunk=128)
            except (ValueError, NotImplementedError) as e:
                print(f"T={T} d={d}: {e}")
                continue
            a = incidence_audit(s)
            print(f"{s.summary()}  exact={a['exact']}  gain={a['pool_gain']:.1f}x  "
                  f"build={s.build_times['total'] * 1e3:.1f}ms")
            for w in s.warnings:
                print(f"    warning: {w}")

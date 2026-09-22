#!/usr/bin/env python3
"""Correctness checks for the revised SMAT mask and its Section 3.3 schedule.

Run this before any timing sweep: a subtly wrong mask produces
plausible-but-meaningless numbers.  Everything here is small and fp64.

    python test_smat.py            # ~2 min on cpu
    python test_smat.py --fast     # skip the slower VC search
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from smat_mask import (build_mask, d_max_geometric, incidence_audit,
                       materialise, planted_hyperplane_normals)
from smat_attn import (SmatDecoder, dense_masked_kernel_attention, featurise,
                       have_triton, make_phi, pool_profiles, apply_incidence,
                       segmented_causal_scan, smat_attention, to_device,
                       walsh_incidence, zeta_transform)

FAILURES = 0


def check(name, ok, detail=""):
    global FAILURES
    FAILURES += not ok
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def banner(s):
    print(f"\n{s}")


# ---------------------------------------------------------------------------
# Theorem 3.1 (i), (ii), (iv) -- structure
# ---------------------------------------------------------------------------

def test_structure():
    banner("Theorem 3.1 (i),(ii),(iv): causal, unit diagonal, <= L_T, Theta(T^2)")
    for T, d in [(256, 1), (512, 2), (1024, 2), (1024, 3), (2048, 3), (2048, 4)]:
        spec = build_mask(T, d, chunk=64)
        M = materialise(spec)
        L = np.tril(np.ones((T, T)))
        causal = bool(np.all(np.triu(M, 1) == 0))
        diag = bool(np.all(np.diag(M) == 1))
        binary = bool(np.all((M == 0) | (M == 1)))
        dominated = bool(np.all(M <= L))
        density = M.sum() / T ** 2
        check(f"T={T} d={d} causal / unit diag / binary / <= L_T",
              causal and diag and binary and dominated,
              f"density={density:.3f}")
        # the two triangular blocks alone give n(n+1)/2 + T_R(T_R+1)/2 ~ T^2/4
        floor = (spec.n * (spec.n + 1) + spec.T_R * (spec.T_R + 1)) / 2
        check(f"T={T} d={d} nnz >= two triangles", M.sum() >= floor,
              f"nnz={int(M.sum()):,} >= {int(floor):,} (= {floor / T**2:.3f} T^2)")

    banner("Remark 3.3: d = 1 is the degenerate member, M^(1) = L_T")
    for T in (256, 512, 1000):
        spec = build_mask(T, 1, chunk=64)
        check(f"T={T} M^(1) == L_T",
              bool(np.array_equal(materialise(spec), np.tril(np.ones((T, T))))))

    banner("the top-left block is L_n, so no landmark is isolated")
    spec = build_mask(1024, 3, chunk=64)
    M = materialise(spec)
    n = spec.n
    rowsum = M[:n, :n].sum(1)
    check("landmark row t attends to exactly t+1 positions",
          bool(np.array_equal(rowsum, np.arange(1, n + 1))),
          f"min={int(rowsum.min())} (was 1 for every row under I_n)")

    banner("range checks")
    try:
        build_mask(1024, 11, chunk=64)
        check("d > floor(log2 T) rejected", False)
    except ValueError:
        check("d > floor(log2 T) rejected", True)
    s = build_mask(1024, 10, chunk=64)
    check("d == floor(log2 T) builds the endpoint mask", s.kind == "endpoint",
          f"kind={s.kind} k={s.k}")
    for T in (4096, 65536):
        dm = d_max_geometric(T, chunk=128)
        check(f"T={T} geometric d_max = {dm} < floor(log2 T) = {T.bit_length()-1}",
              dm < T.bit_length() - 1)


def test_block_layout():
    """The dense mask must agree with Eq. (3.9) block by block."""
    banner("block layout matches Eq. (3.9), and G = R C S^T")
    for T, d, nP in [(1024, 2, 0), (2048, 3, 0), (2048, 4, 0), (2048, 3, 16)]:
        spec = build_mask(T, d, chunk=64, n_P=nP)
        M = materialise(spec)
        n, T_R = spec.n, spec.T_R
        ok_L = bool(np.array_equal(M[:n, :n], np.tril(np.ones((n, n)))))
        ok_0 = bool(np.all(M[:n, n:] == 0))
        ok_R = bool(np.array_equal(M[n:, n:], np.tril(np.ones((T_R, T_R)))))
        ok_P = bool(np.all(M[n:, :nP] == 1)) if nP else True
        check(f"T={T} d={d} |P|={nP}: L_n / 0 / L_TR / broadcast", 
              ok_L and ok_0 and ok_R and ok_P)

        C = spec.dense_C()
        prof = spec.prof_of_column()
        rng = np.random.default_rng(0)
        ok_rows = True
        for i in rng.integers(0, T_R, size=24):
            want = np.flatnonzero(C[i % spec.B, prof]) + spec.n_P
            got = np.flatnonzero(M[n + i, spec.n_P:n]) + spec.n_P
            ok_rows &= bool(np.array_equal(want, got))
        check(f"T={T} d={d} row n+i on X == H_(i mod B)", ok_rows)

        # G factorises as R C S^T exactly
        R = np.zeros((T_R, spec.B)); R[np.arange(T_R), np.arange(T_R) % spec.B] = 1
        S = np.zeros((spec.n_X, spec.N0)); S[np.arange(spec.n_X), prof] = 1
        check(f"T={T} d={d} G == R C S^T",
              bool(np.array_equal(M[n:, spec.n_P:n], R @ C @ S.T)))


def test_planted():
    banner("Eq. (3.5): the planted hyperplanes satisfy  e_i in H_R <=> i in R")
    for T, d in [(1024, 2), (2048, 3), (4096, 4), (16384, 5)]:
        spec = build_mask(T, d, chunk=64)
        a = incidence_audit(spec)
        check(f"T={T} d={d} planted identity over all 2^(d-1) types",
              bool(a["planted_identity"]),
              f"B={spec.B} >= 2^(d-1)={1 << (d-1)}: {a['B_ge_2_pow_d_minus_1']}, "
              f"T_R >= 2B: {a['T_R_ge_2B']}")
        # the normals really are legitimate (nonzero) and pairwise distinct
        hs = planted_hyperplane_normals(d)
        check(f"T={T} d={d} normals nonzero and pairs distinct",
              all(np.any(a_ != 0) for a_, _ in hs)
              and len({(tuple(a_), b) for a_, b in hs}) == len(hs))


def test_incidence():
    banner("counting: nnz(C) = sum_x nu(x), deg(H) = q^(d-2), I_tok / I_prof ~ q")
    for T, d in [(4096, 2), (16384, 2), (4096, 3), (16384, 3), (65536, 3),
                 (16384, 4), (65536, 4)]:
        spec = build_mask(T, d, chunk=128)
        a = incidence_audit(spec)
        check(f"T={T} d={d}", bool(a["exact"]),
              f"I_prof={a['I_prof']:,} I_tok={a['I_tok']:,} "
              f"gain={a['pool_gain']:.1f}x (q={spec.q})")

    banner("exact algebra: I_prof = B q^(d-2), so log I_prof / log q = 2d-3")
    for d in (2, 3, 4, 5):
        Ts = [2 ** k for k in range(14, 20)]
        specs = [s for s in (build_mask(T, d, chunk=128) for T in Ts)]
        exact = all(s.I_prof == s.B * s.q ** (d - 2) for s in specs)
        check(f"d={d} I_prof == B q^(d-2) at every T", exact)
        # closed form: B = q(q^{d-1}-1)/(q-1), so I_prof = q^{d-1}(q^{d-1}-1)/(q-1),
        # which is q^{2d-3} up to a factor 1 + O(1/q) -- the constant the
        # Theta(T^{2-3/d}) of Eq. (3.21) hides.
        closed = all(s.I_prof == s.q ** (d - 1) * (s.q ** (d - 1) - 1) // (s.q - 1)
                     for s in specs)
        ratios = [s.I_prof / s.q ** (2 * d - 3) for s in specs]
        check(f"d={d} I_prof == q^(d-1)(q^(d-1)-1)/(q-1) exactly", closed,
              f"I_prof / q^(2d-3) in [{min(ratios):.4f}, {max(ratios):.4f}]")

    banner("fitted exponents in T (Table 1).  q = nextprime(n^(1/d)) moves in")
    banner("coarse steps, so these are staircases and converge from below.")
    for d in (2, 3, 4):
        Ts = [2 ** k for k in range(14, 20)]
        specs = [build_mask(T, d, chunk=128) for T in Ts]
        for key, pred in [("I_prof", 2 - 3 / d), ("I_tok", 2 - 2 / d)]:
            ys = np.array([getattr(s, key) for s in specs], float)
            alpha = np.polyfit(np.log(Ts), np.log(ys), 1)[0]
            check(f"d={d} {key} exponent ~ {pred:.3f}",
                  abs(alpha - pred) < 0.25, f"alpha={alpha:.3f}")


# ---------------------------------------------------------------------------
# Theorem 3.1 (iii) -- exact VC dimension
# ---------------------------------------------------------------------------

def test_vc_dimension(fast=False):
    banner("Theorem 3.1 (iii): exact VC dimension of the materialised mask")
    from vc_dimension import pseudo_dimension_detailed

    cases = [(256, 1), (256, 2), (512, 2), (512, 3), (1024, 3)]
    if not fast:
        cases += [(2048, 3), (4096, 4), (8192, 4)]
    for T, d in cases:
        spec = build_mask(T, d, chunk=64)
        M = materialise(spec, np.int8)
        res = pseudo_dimension_detailed(M, points="columns", time_limit=90.0)
        check(f"T={T} d={d}: VC = {res.dimension} (exact={res.exact})",
              res.dimension == d and res.exact,
              f"ceiling floor(log2 T) = {T.bit_length() - 1}")

    banner("endpoint d = floor(log2 T): zeta landmarks + Walsh rows")
    for T in (128, 256, 255, 300):
        d = T.bit_length() - 1
        spec = build_mask(T, d, chunk=8)
        M = materialise(spec, np.int8)
        res = pseudo_dimension_detailed(M, points="columns", time_limit=120.0)
        need = spec.T_R >= spec.B + 1
        check(f"T={T} d=floor(log2 T)={d}: VC = {res.dimension}"
              f"{'' if need else '  [T_R < B+1]'}",
              res.dimension == d if need else res.dimension >= d - 1,
              f"n=2^{spec.k}={spec.n} B={spec.B} T_R={spec.T_R}, exact={res.exact}")


# ---------------------------------------------------------------------------
# Section 3.3 -- the schedule
# ---------------------------------------------------------------------------

def _rand(T, nb=2, d_qk=16, d_v=12, r=20, dt=torch.float64, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(nb, T, d_qk, dtype=dt)
    K = torch.randn(nb, T, d_qk, dtype=dt)
    V = torch.randn(nb, T, d_v, dtype=dt)
    g = torch.rand(nb, T, dtype=dt) + 0.5
    h = torch.rand(nb, T, dtype=dt) + 0.5
    phi = make_phi(d_qk, r, dtype=dt)
    return featurise(Q, K, V, phi, g=g, h=h), g


def test_equivalence():
    banner("Theorem 3.2 exactness: Algorithm 1 == dense masked attention (fp64)")
    for T, d, nP in [(512, 1, 0), (512, 2, 0), (1024, 2, 0), (1024, 3, 0),
                     (2048, 3, 0), (2048, 4, 0), (1024, 3, 16), (1024, 10, 0)]:
        spec = to_device(build_mask(T, d, chunk=64, n_P=nP), "cpu")
        (Phi, Psi, Vb), g = _rand(T)
        M = torch.from_numpy(materialise(spec)).double()
        ref = dense_masked_kernel_attention(Phi, Psi, Vb, M) * g.unsqueeze(-1)
        got = smat_attention(Phi, Psi, Vb, spec, chunk=64, g=g)
        err = (got - ref).abs().max().item()
        check(f"T={T} d={d} |P|={nP} kind={spec.kind}: max|smat - dense|",
              err < 1e-10, f"{err:.2e}")


def test_backends():
    banner("pooled == direct == subtractive (all three realise the same mask)")
    for T, d in [(1024, 2), (2048, 3), (2048, 4)]:
        spec = to_device(build_mask(T, d, chunk=64), "cpu", want_direct=True)
        (Phi, Psi, Vb), g = _rand(T)
        base = smat_attention(Phi, Psi, Vb, spec, chunk=64, g=g)
        for lr in ("direct", "subtractive"):
            got = smat_attention(Phi, Psi, Vb, spec, chunk=64, g=g, long_range=lr)
            err = (got - base).abs().max().item()
            check(f"T={T} d={d} {lr}", err < 1e-10, f"{err:.2e}")


def test_scan_and_transforms():
    banner("the segmented scan has exactly one reset, at n")
    T, d = 1024, 3
    spec = build_mask(T, d, chunk=64)
    (Phi, Psi, Vb), _ = _rand(T)
    got = segmented_causal_scan(Phi, Psi, Vb, n=spec.n, chunk=64)
    Mseg = np.zeros((T, T))
    Mseg[:spec.n, :spec.n] = np.tril(np.ones((spec.n, spec.n)))
    Mseg[spec.n:, spec.n:] = np.tril(np.ones((spec.T_R, spec.T_R)))
    ref = torch.einsum("nts,nsv->ntv",
                       torch.einsum("nta,nsa->nts", Phi, Psi)
                       * torch.from_numpy(Mseg), Vb)
    check("scan == block-diagonal causal reference",
          (got - ref).abs().max().item() < 1e-10,
          f"{(got - ref).abs().max().item():.2e}")

    banner("chunk-width invariance")
    spec = to_device(build_mask(1024, 2, chunk=32), "cpu")
    (Phi, Psi, Vb), _ = _rand(1024)
    base = smat_attention(Phi, Psi, Vb, spec, chunk=32)
    for C in (64, 128, 256):
        s2 = to_device(build_mask(1024, 2, chunk=C), "cpu")
        # a different chunk moves n, so compare against that spec's own dense mask
        M = torch.from_numpy(materialise(s2)).double()
        ref = dense_masked_kernel_attention(Phi, Psi, Vb, M)
        err = (smat_attention(Phi, Psi, Vb, s2, chunk=C) - ref).abs().max().item()
        check(f"C={C} (n={s2.n})", err < 1e-10, f"{err:.2e}")

    banner("butterfly transforms: zeta = L_2^(x)k, Walsh route == dense C")
    nb, N, r, p = 2, 64, 5, 4
    F = torch.randn(nb, N, r, p, dtype=torch.float64)
    a = np.arange(N)
    Z = torch.from_numpy(((a[:, None] & a[None, :]) == a[None, :]).astype(float))
    ref = torch.einsum("ax,nxrp->narp", Z, F)
    check("zeta_transform == dense Z_k",
          (zeta_transform(F) - ref).abs().max().item() < 1e-12)
    Cw = torch.from_numpy(
        ((np.bitwise_count(a[:, None] & a[None, :]) % 2) == 0).astype(float))
    refw = torch.einsum("hx,nxrp->nhrp", Cw, F)
    check("walsh_incidence == dense Walsh C",
          (walsh_incidence(F) - refw).abs().max().item() < 1e-11)


def test_exact_noninfluence():
    """A masked-out column must not move a row's output *at all*, not merely
    to tolerance.

    The rest of the suite checks values against a dense reference to ~1e-10,
    which a segment reset built as "global cumsum minus the landmark total"
    passes comfortably while still routing the whole landmark block's mass
    through every recent chunk's prefix, where it cancels only to O(eps).  The
    expressivity claim of Sec. 4 is that unrequested items are *exactly* non-
    influential because the mask removed their edges, so it needs an exact test:
    perturb a column and require the rows that exclude it to be bit-identical.
    """
    banner("masked-out columns are exactly non-influential (bitwise)")
    for T, d, C in ((1024, 3, 128), (1024, 2, 64), (512, 1, 64)):
        spec_np = build_mask(T, d, chunk=C)
        spec = to_device(spec_np, "cpu")
        M = materialise(spec_np).astype(bool)
        (Phi, Psi, Vb), _ = _rand(T)
        base = smat_attention(Phi, Psi, Vb, spec, chunk=C)

        rng = np.random.default_rng(0)
        worst, bad = 0.0, 0
        for j in rng.choice(T, size=6, replace=False):
            Psi2, Vb2 = Psi.clone(), Vb.clone()
            Psi2[:, int(j)] = torch.randn_like(Psi2[:, int(j)]).abs()
            Vb2[:, int(j)] = torch.randn_like(Vb2[:, int(j)])
            out = smat_attention(Phi, Psi2, Vb2, spec, chunk=C)
            excl = ~M[:, int(j)]                      # rows that cannot see j
            delta = (out[:, excl] - base[:, excl]).abs().max().item()
            worst = max(worst, delta)
            bad += delta != 0.0
        check(f"T={T} d={d}: rows excluding a perturbed column are bit-identical",
              bad == 0, f"{bad}/6 columns moved excluded rows, worst {worst:.2e}")


def test_decode():
    banner("Remark 3.9: streaming decode reproduces the prefill rows")
    for T, d in [(512, 2), (1024, 3), (2048, 4), (512, 1)]:
        spec = to_device(build_mask(T, d, chunk=64), "cpu")
        (Phi, Psi, Vb), _ = _rand(T)
        full = smat_attention(Phi, Psi, Vb, spec, chunk=64)
        n = spec.n
        dec = SmatDecoder(Psi[:, :n], Vb[:, :n], spec)
        errs = [(dec.step(Phi[:, i], Psi[:, i], Vb[:, i])
                 - full[:, i]).abs().max().item() for i in range(n, T)]
        check(f"T={T} d={d}: max over {T - n} decoded tokens",
              max(errs) < 1e-11, f"{max(errs):.2e}")
        words = dec.cache_words()
        want = Psi.shape[0] * (spec.B + 1) * Psi.shape[-1] * Vb.shape[-1]
        check(f"T={T} d={d}: cache == (B+1) r p words", words == want,
              f"{words:,} vs KV {SmatDecoder.kv_cache_words(2, T, 12):,}")


def test_dtype_paths():
    banner("reduced precision stays close to fp64")
    T, d = 1024, 3
    spec = to_device(build_mask(T, d, chunk=128), "cpu")
    nb, d_qk, d_v, r = 1, 32, 32, 32
    torch.manual_seed(4)
    Q = torch.randn(nb, T, d_qk, dtype=torch.float64)
    K = torch.randn(nb, T, d_qk, dtype=torch.float64)
    V = torch.randn(nb, T, d_v, dtype=torch.float64)
    ref = smat_attention(*featurise(Q, K, V, make_phi(d_qk, r, dtype=torch.float64)),
                         spec, chunk=128)
    for dt, tol in [(torch.float32, 1e-4), (torch.bfloat16, 6e-2)]:
        phi = make_phi(d_qk, r, dtype=dt, )
        Phi, Psi, Vb = featurise(Q.to(dt), K.to(dt), V.to(dt), phi)
        got = smat_attention(Phi, Psi, Vb, spec, chunk=128,
                             acc_dtype=torch.float32)
        err = (got.double() - ref).abs().max().item()
        check(f"{dt} vs fp64", err < tol, f"max err {err:.2e}")


def test_triton():
    banner("triton kernels agree with the torch path")
    if not have_triton():
        print("  (no cuda/triton here -- run this on the GPU node)")
        return
    dev = "cuda"
    # The fused kernels use TF32 for fp32 inputs, as the benchmark's torch path
    # does; match that here so the comparison is like for like.
    torch.backends.cuda.matmul.allow_tf32 = True
    for T, d, dt in [(4096, 2, torch.float32), (4096, 3, torch.float32),
                     (8192, 3, torch.bfloat16), (8192, 4, torch.bfloat16)]:
        spec = to_device(build_mask(T, d, chunk=128), dev)
        torch.manual_seed(0)
        nb, d_qk, d_v, r = 2, 32, 32, 32
        Q = torch.randn(nb, T, d_qk, device=dev, dtype=dt)
        K = torch.randn(nb, T, d_qk, device=dev, dtype=dt)
        V = torch.randn(nb, T, d_v, device=dev, dtype=dt)
        Phi, Psi, Vb = featurise(Q, K, V, make_phi(d_qk, r, device=dev, dtype=dt))
        ref = smat_attention(Phi, Psi, Vb, spec, chunk=128,
                             scan_backend="torch", incidence_backend="torch",
                             acc_dtype=torch.float32)
        got = smat_attention(Phi, Psi, Vb, spec, chunk=128,
                             scan_backend="triton", incidence_backend="triton",
                             acc_dtype=torch.float32)
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        tol = 2e-3 if dt == torch.float32 else 3e-2
        check(f"T={T} d={d} {dt}: triton vs torch (rel)", rel < tol, f"{rel:.2e}")

        Fa = pool_profiles(Psi, Vb, spec, acc_dtype=torch.float32,
                           backend="torch")
        Fb = pool_profiles(Psi, Vb, spec, acc_dtype=torch.float32,
                           backend="triton")
        r_ = ((Fa - Fb).abs().max() / Fa.abs().max()).item()
        check(f"T={T} d={d}: pooling kernel", r_ < tol, f"{r_:.2e}")
        a = apply_incidence(Fa, spec, backend="torch")
        b = apply_incidence(Fa, spec, backend="triton")
        r_ = ((a - b).abs().max() / a.abs().max()).item()
        check(f"T={T} d={d}: incidence kernel", r_ < 1e-5, f"{r_:.2e}")

        # the fp64 reference path must never be routed through the kernels
        Ph64 = Phi.double()
        F64 = pool_profiles(Ph64, Vb.double(), spec, acc_dtype=torch.float64)
        check(f"T={T} d={d}: fp64 path stays in fp64",
              F64.dtype == torch.float64, str(F64.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-vc", action="store_true")
    args = ap.parse_args()

    test_structure()
    test_block_layout()
    test_planted()
    test_incidence()
    if not args.skip_vc:
        test_vc_dimension(fast=args.fast)
    test_equivalence()
    test_backends()
    test_scan_and_transforms()
    test_exact_noninfluence()
    test_decode()
    test_dtype_paths()
    test_triton()
    print(f"\n{'FAILURES: ' + str(FAILURES) if FAILURES else 'all tests passed'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

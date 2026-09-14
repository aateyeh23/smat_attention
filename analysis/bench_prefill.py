#!/usr/bin/env python3
"""Timing harness for the revised SMAT schedule.

The deliverable is a *fitted exponent*, not a point speedup: we sweep ``T``, fit
``log(time) = alpha log T + beta``, and report ``alpha`` against the prediction.
Two exponents are claimed by Sec. 3.3 and both are measured here:

  pooled long range    alpha -> 2 - 3/d   (Eq. 3.21, the schedule of the draft)
  direct long range    alpha -> 2 - 2/d   (the unpooled comparison)

and total prefill should be *linear* in ``T`` for ``d <= 3``, because the
long-range term is then dominated by the Theta(T) scan.

    python bench_smat.py --quick --device cpu --threads 1
    python bench_smat.py --device cuda --dtype bf16 --out results_bf16.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from smat.mask import build_mask, estimate_cost, incidence_audit, materialise
from smat.attention import (SmatDecoder, dense_masked_kernel_attention, featurise,
                       have_triton, make_phi, smat_attention, to_device)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16,
          "fp16": torch.float16, "fp64": torch.float64}


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class PhaseTimer:
    """``with timer("name"):`` attribution, CUDA-event based on GPU."""

    def __init__(self, device):
        self.device = device
        self.spans: List[tuple] = []

    @contextmanager
    def __call__(self, name):
        if self.device.type == "cuda":
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record(); yield; b.record()
            self.spans.append((name, a, b))
        else:
            t0 = time.perf_counter(); yield
            self.spans.append((name, t0, time.perf_counter()))

    def results(self) -> Dict[str, float]:
        sync(self.device)
        out: Dict[str, float] = {}
        for name, a, b in self.spans:
            ms = a.elapsed_time(b) if self.device.type == "cuda" else (b - a) * 1e3
            out[name] = out.get(name, 0.0) + ms
        return out


def bench(fn, device, warmup: int = 3, repeat: int = 10) -> float:
    """Median wall-clock of ``fn`` in ms."""
    for _ in range(warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(repeat):
        if device.type == "cuda":
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); b.synchronize()
            ts.append(a.elapsed_time(b))
        else:
            t0 = time.perf_counter(); fn()
            ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def check_cpu_health():
    a, b = torch.randn(4, 1, 32), torch.randn(4, 32, 33)
    for _ in range(50):
        a @ b
    t0 = time.perf_counter()
    for _ in range(200):
        a @ b
    us = (time.perf_counter() - t0) / 200 * 1e6
    if us > 200:
        print(f"WARNING: a tiny matmul takes {us:.0f}us with "
              f"{torch.get_num_threads()} threads -- this node's thread pool is "
              f"contended and every timing here is noise.  Use --threads 1.")
    else:
        print(f"cpu check: small matmul {us:.1f}us, "
              f"{torch.get_num_threads()} threads")


def _peak(device, fn):
    if device.type != "cuda":
        fn(); return float("nan")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn(); torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 2 ** 20


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------

def run_config(T: int, d: int, args, device, dtype) -> Optional[Dict]:
    est = estimate_cost(T, d, args.r, args.d_v, chunk=args.chunk, n_P=args.n_P)
    if not est.get("feasible", True):
        print(f"  T={T:>7} d={d}: skipped ({est['reason']})")
        return None
    if est["I_prof"] > args.max_I_prof:
        print(f"  T={T:>7} d={d}: skipped (nnz(C)={est['I_prof']:,} > "
              f"--max-I-prof {args.max_I_prof:,})")
        return None

    row: Dict[str, object] = dict(
        T=T, d=d, nb=args.nb, r=args.r, d_qk=args.d_qk, d_v=args.d_v,
        chunk=args.chunk, dtype=args.dtype, device=device.type,
        triton=bool(have_triton() and device.type == "cuda"))

    # -- build + preprocessing --------------------------------------------
    t0 = time.perf_counter()
    spec = build_mask(T, d, chunk=args.chunk, n_P=args.n_P)
    row["build_ms"] = (time.perf_counter() - t0) * 1e3
    for k, v in spec.build_times.items():
        row[f"build_{k}_ms"] = v * 1e3
    for w in spec.warnings:
        print(f"    warning: {w}")

    want_direct = args.direct and spec.I_tok <= args.max_I_tok
    t0 = time.perf_counter()
    to_device(spec, device, want_direct=want_direct)
    sync(device)
    row["stage_ms"] = (time.perf_counter() - t0) * 1e3

    row.update(kind=spec.kind, q=spec.q, B=spec.B, N0=spec.N0, n=spec.n,
               T_R=spec.T_R, I_prof=spec.I_prof, I_tok=spec.I_tok,
               deg=spec.uniform_deg)
    a = incidence_audit(spec)
    row["incidence_exact"] = a["exact"]
    row["pool_gain"] = a["pool_gain"]

    # -- tensors -----------------------------------------------------------
    torch.manual_seed(args.seed)
    Q = torch.randn(args.nb, T, args.d_qk, device=device, dtype=dtype)
    K = torch.randn(args.nb, T, args.d_qk, device=device, dtype=dtype)
    V = torch.randn(args.nb, T, args.d_v, device=device, dtype=dtype)
    g = torch.rand(args.nb, T, device=device, dtype=dtype) + 0.5
    h = torch.rand(args.nb, T, device=device, dtype=dtype) + 0.5
    phi = make_phi(args.d_qk, args.r, device=device, dtype=dtype, seed=args.seed)
    row["featurise_ms"] = bench(lambda: featurise(Q, K, V, phi, g=g, h=h),
                                device, args.warmup, args.repeat)
    Phi, Psi, Vb = featurise(Q, K, V, phi, g=g, h=h)

    def prefill(lr="pooled", sb="auto", ib="auto"):
        return lambda: smat_attention(Phi, Psi, Vb, spec, chunk=args.chunk, g=g,
                                      long_range=lr, scan_backend=sb,
                                      incidence_backend=ib,
                                      acc_dtype=torch.float32)

    # -- prefill: the schedule of the draft --------------------------------
    row["prefill_ms"] = bench(prefill(), device, args.warmup, args.repeat)
    row["prefill_peak_MB"] = _peak(device, prefill())

    timer = PhaseTimer(device)
    smat_attention(Phi, Psi, Vb, spec, chunk=args.chunk, g=g,
                   acc_dtype=torch.float32, timer=timer)
    for k, v in timer.results().items():
        row[f"phase_{k}_ms"] = v
    row["phase_lr_total_ms"] = sum(v for k, v in timer.results().items()
                                   if k.startswith("lr_"))

    # -- the three implementation comparisons of Sec. 3.3 ------------------
    # The torch incidence path streams the whole of F once per degree step, so
    # at a large nnz(C) it is minutes per call -- not a useful comparison point
    # and not worth the wall clock.  Compare where the comparison is meaningful.
    if (device.type == "cuda" and have_triton()
            and spec.I_prof <= args.max_I_prof_torch):
        row["prefill_torch_ms"] = bench(prefill(sb="torch", ib="torch"),
                                        device, args.warmup, args.repeat)
        row["prefill_torch_peak_MB"] = _peak(device, prefill(sb="torch", ib="torch"))
        t2 = PhaseTimer(device)
        smat_attention(Phi, Psi, Vb, spec, chunk=args.chunk, g=g,
                       scan_backend="torch", incidence_backend="torch",
                       acc_dtype=torch.float32, timer=t2)
        for k, v in t2.results().items():
            row[f"torchphase_{k}_ms"] = v
    if want_direct:
        row["prefill_direct_ms"] = bench(prefill(lr="direct"), device,
                                         args.warmup, args.repeat)
        t3 = PhaseTimer(device)
        smat_attention(Phi, Psi, Vb, spec, chunk=args.chunk, g=g,
                       long_range="direct", acc_dtype=torch.float32, timer=t3)
        row["phase_direct_incidence_ms"] = t3.results().get("lr_incidence", "")
    if args.subtractive:
        row["prefill_subtractive_ms"] = bench(prefill(lr="subtractive"), device,
                                              args.warmup, args.repeat)

    # -- Remark 3.10: does the subtractive route lose the normalizer? ------
    if args.subtractive and T <= args.numeric_max_T:
        row.update(_normalizer_conditioning(Q, K, V, g, h, spec, args, device))

    # -- decode ------------------------------------------------------------
    n = spec.n
    dec = SmatDecoder(Psi[:, :n], Vb[:, :n], spec, acc_dtype=torch.float32)
    Pt, St, Vt = Phi[:, n], Psi[:, n], Vb[:, n]

    def step_many():
        for _ in range(args.decode_steps):
            dec.step(Pt, St, Vt)

    row["decode_us_per_token"] = (bench(step_many, device, 1, 3)
                                  * 1e3 / args.decode_steps)
    row["cache_words"] = dec.cache_words()
    row["kv_cache_words"] = SmatDecoder.kv_cache_words(args.nb, T, args.d_v)
    row["cache_ratio"] = row["cache_words"] / row["kv_cache_words"]

    # -- baselines ---------------------------------------------------------
    row["sdpa_ms"] = ""
    if T <= args.baseline_max_T:
        q4, k4, v4 = Q.unsqueeze(1), K.unsqueeze(1), V.unsqueeze(1)
        try:
            row["sdpa_ms"] = bench(
                lambda: F.scaled_dot_product_attention(q4, k4, v4, is_causal=True),
                device, args.warmup, args.repeat)
        except RuntimeError as e:
            print(f"    sdpa failed: {e}")
    row["dense_masked_ms"] = ""
    if T <= args.dense_max_T:
        Md = torch.from_numpy(materialise(spec, np.float32)).to(device, dtype)
        row["dense_masked_ms"] = bench(
            lambda: dense_masked_kernel_attention(Phi, Psi, Vb, Md),
            device, 1, max(3, args.repeat // 3))
        del Md

    line = (f"  T={T:>7} d={d}  build {row['build_ms']:7.1f}ms  "
            f"prefill {row['prefill_ms']:8.2f}ms  "
            f"lr {row['phase_lr_total_ms']:7.2f}ms  "
            f"decode {row['decode_us_per_token']:6.1f}us/tok  "
            f"cache {row['cache_ratio']:6.3f}xKV")
    if row["sdpa_ms"] != "":
        line += f"  sdpa {row['sdpa_ms']:8.2f}ms"
    print(line, flush=True)
    return row


def _normalizer_conditioning(Q, K, V, g, h, spec, args, device) -> Dict:
    """Remark 3.10, measured on the quantity the remark is about.

    Both routes compute the same long-range normalizer for recent row ``i``:

        additive      a = Phi_i . U_{rho(i)}
        subtractive   b = Phi_i . F_tot  -  Phi_i . Ubar_{rho(i)}

    They are algebraically equal, but ``a`` is a sum of nonnegative terms while
    ``b`` is a difference of two larger ones.  Each hyperplane carries
    ``deg / N0 = 1/q`` of the profile mass, so the retained fraction is ~1/q and
    a rounding error in the large quantity is amplified by ~q in the small one.
    Inputs are held fixed and only the accumulation dtype varies, so input
    quantisation cannot mask the effect.
    """
    from smat_attn import pool_profiles, apply_incidence, query_by_type
    out: Dict[str, object] = {}
    phid = make_phi(args.d_qk, args.r, device=device, dtype=torch.float64,
                    seed=args.seed)
    Ph, Ps, Vp = featurise(Q.double(), K.double(), V.double(), phid,
                           g=g.double(), h=h.double())
    F64 = pool_profiles(Ps, Vp, spec, acc_dtype=torch.float64)
    U64 = apply_incidence(F64, spec, backend="torch")
    kept = query_by_type(Ph[:, spec.n:], U64, spec)[..., -1]
    large = query_by_type(Ph[:, spec.n:],
                          F64.sum(1, keepdim=True).expand_as(U64), spec)[..., -1]
    out["retained_fraction"] = (kept.abs().mean() / large.abs().mean()).item()
    out["retained_predicted"] = (spec.uniform_deg / spec.N0) if spec.N0 else 1.0

    for name, acc in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        Fa = F64.to(acc)
        Ua = apply_incidence(Fa.float(), spec, backend="torch").to(acc)
        a = query_by_type(Ph[:, spec.n:].to(acc), Ua, spec)[..., -1].double()
        tot = Fa.sum(1, keepdim=True).expand_as(Ua)
        b = (query_by_type(Ph[:, spec.n:].to(acc), tot, spec)[..., -1]
             - query_by_type(Ph[:, spec.n:].to(acc), (tot - Ua).contiguous(),
                             spec)[..., -1]).double()
        den = kept.abs().mean()
        out[f"relerr_additive_{name}"] = ((a - kept).abs().mean() / den).item()
        out[f"relerr_subtractive_{name}"] = ((b - kept).abs().mean() / den).item()
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def fit_exponent(rows, d, key="prefill_ms", min_T=0):
    pts = [(r["T"], r[key]) for r in rows
           if r["d"] == d and r["T"] >= min_T and r.get(key) not in ("", None)
           and not (isinstance(r.get(key), float) and np.isnan(r[key]))]
    if len(pts) < 3:
        return None
    Ts = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    alpha, beta = np.polyfit(np.log(Ts), np.log(ys), 1)
    pred = np.exp(beta) * Ts ** alpha
    r2 = 1 - ((ys - pred) ** 2).sum() / max(1e-30, ((ys - ys.mean()) ** 2).sum())
    return float(alpha), float(beta), float(r2), len(pts)


def crossover(rows, d, key="prefill_ms"):
    pts = sorted((r["T"], r[key], r["sdpa_ms"]) for r in rows
                 if r["d"] == d and r["sdpa_ms"] != "" and r.get(key) not in ("", None))
    for T, smat, sdpa in pts:
        if smat < sdpa:
            return float(T)
    return None


def write_rows(path, rows, info=None):
    """Write the CSV.  Called after *every* configuration, not just at the end.

    A sweep that reaches its wall clock one config short should still hand back
    everything it measured; losing hours of GPU time to a trailing write is not
    a trade worth making.
    """
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if info is not None:
        with open(path.replace(".csv", "_config.json"), "w") as f:
            json.dump(info, f, indent=2, default=str)


def report(rows, args):
    ds = sorted({r["d"] for r in rows})
    print("\n" + "=" * 78)
    print("FITTED EXPONENTS   log(ms) = alpha log T + beta")
    print("=" * 78)
    print(f"{'d':>3} {'quantity':>26} {'alpha':>8} {'predicted':>10} {'R^2':>7} {'pts':>4}")
    for d in ds:
        for key, pred, label in [
                ("phase_lr_incidence_ms", 2 - 3 / d if d > 1 else 0.0,
                 "incidence C  (pooled)"),
                ("phase_direct_incidence_ms", 2 - 2 / d if d > 1 else 0.0,
                 "incidence C  (direct)"),
                ("phase_lr_pool_ms", 1.0, "profile pooling S^T"),
                ("phase_lr_query_ms", 1.0, "type-major GEMM R"),
                ("phase_scan_ms", 1.0, "causal scan"),
                ("prefill_ms", None, "total prefill"),
                ("sdpa_ms", 2.0, "causal SDPA")]:
            fit = fit_exponent(rows, d, key=key, min_T=args.fit_min_T)
            if fit is None:
                continue
            alpha, _, r2, npts = fit
            ps = "--" if pred is None else f"{pred:.3f}"
            print(f"{d:>3} {label:>26} {alpha:8.3f} {ps:>10} {r2:7.4f} {npts:>4}")
    print("\nOnly the incidence phase carries the T^{2-3/d} exponent: pooling and")
    print("the type-major GEMM are Theta(T) at every d, and total prefill is the")
    print("blend, so for d <= 3 it should come out ~1.  A phase whose absolute")
    print("time is at the kernel-launch floor fits alpha ~ 0 regardless; the")
    print("achieved-rate column below says when that is happening.")
    print(f"\n{'T':>9} {'d':>3} {'I_prof':>11} {'incid ms':>9} "
          f"{'Gadd/s':>9}   (below ~10 GFLOP/s the phase is launch-bound)")
    for r in sorted(rows, key=lambda r: (r["d"], r["T"])):
        ms = r.get("phase_lr_incidence_ms")
        if not ms:
            continue
        rate = r["nb"] * r["I_prof"] * r["r"] * (r["d_v"] + 1) / (ms * 1e-3) / 1e9
        print(f"{r['T']:>9,} {r['d']:>3} {r['I_prof']:>11,} {ms:9.3f} {rate:9.2f}")

    print("\n" + "=" * 78)
    print("CROSSOVER T* vs causal scaled_dot_product_attention")
    print("=" * 78)
    for d in ds:
        x = crossover(rows, d)
        print(f"  d={d}: " + (f"T* <= {int(x):,} (first measured T where SMAT wins)"
                              if x else "no measured crossover in this sweep"))

    if any("prefill_torch_ms" in r for r in rows):
        print("\n" + "=" * 78)
        print("TRITON vs TORCH  (same schedule, two realisations)")
        print("=" * 78)
        print(f"{'T':>9} {'d':>3} {'triton ms':>10} {'torch ms':>10} {'speedup':>8} "
              f"{'peak MB':>9} {'torch MB':>9}")
        for r in sorted(rows, key=lambda r: (r["d"], r["T"])):
            if "prefill_torch_ms" not in r:
                continue
            print(f"{r['T']:>9,} {r['d']:>3} {r['prefill_ms']:10.2f} "
                  f"{r['prefill_torch_ms']:10.2f} "
                  f"{r['prefill_torch_ms'] / r['prefill_ms']:8.2f}x "
                  f"{r['prefill_peak_MB']:9.1f} {r['prefill_torch_peak_MB']:9.1f}")

    if any("prefill_direct_ms" in r for r in rows):
        print("\n" + "=" * 78)
        print("POOLED vs DIRECT long range   (Theta(T^{2-3/d}) vs Theta(T^{2-2/d}))")
        print("=" * 78)
        print(f"{'T':>9} {'d':>3} {'I_prof':>12} {'I_tok':>13} {'pooled ms':>10} "
              f"{'direct ms':>10} {'ratio':>7}")
        for r in sorted(rows, key=lambda r: (r["d"], r["T"])):
            if "phase_direct_incidence_ms" not in r or r["phase_direct_incidence_ms"] == "":
                continue
            print(f"{r['T']:>9,} {r['d']:>3} {r['I_prof']:>12,} {r['I_tok']:>13,} "
                  f"{r['phase_lr_incidence_ms']:10.3f} "
                  f"{r['phase_direct_incidence_ms']:10.3f} "
                  f"{r['phase_direct_incidence_ms'] / max(1e-9, r['phase_lr_incidence_ms']):6.2f}x")

    if any("relerr_subtractive_fp32" in r for r in rows):
        print("\n" + "=" * 78)
        print("REMARK 3.10  the subtractive alternative, measured")
        print("=" * 78)
        print(f"{'T':>9} {'d':>3} {'q':>5} {'kept/total':>11} {'1/q':>7} "
              f"{'add fp32':>10} {'sub fp32':>10} {'add bf16':>10} "
              f"{'sub bf16':>10} {'amp':>7}")
        for r in sorted(rows, key=lambda r: (r["d"], r["T"])):
            if "relerr_subtractive_fp32" not in r:
                continue
            amp = (r["relerr_subtractive_fp32"]
                   / max(1e-30, r["relerr_additive_fp32"]))
            print(f"{r['T']:>9,} {r['d']:>3} {r['q']:>5} "
                  f"{r['retained_fraction']:11.4f} "
                  f"{1.0 / max(1, r['q']):7.4f} "
                  f"{r['relerr_additive_fp32']:10.2e} "
                  f"{r['relerr_subtractive_fp32']:10.2e} "
                  f"{r['relerr_additive_bf16']:10.2e} "
                  f"{r['relerr_subtractive_bf16']:10.2e} {amp:6.1f}x")

    print("\n" + "=" * 78)
    print("DECODE: cost per token and cache size vs a softmax KV cache")
    print("=" * 78)
    print("Per-token cost is a handful of small kernel launches, so the absolute")
    print("number is launch-bound, not arithmetic-bound; what is being verified")
    print("here is that it does not grow with T.")
    print(f"{'T':>9} {'d':>3} {'us/token':>10} {'cache words':>14} "
          f"{'KV words':>14} {'ratio':>8}")
    for r in sorted(rows, key=lambda r: (r["d"], r["T"])):
        print(f"{r['T']:>9,} {r['d']:>3} {r['decode_us_per_token']:10.2f} "
              f"{r['cache_words']:>14,} {r['kv_cache_words']:>14,} "
              f"{r['cache_ratio']:8.3f}")

    print("\n" + "=" * 78)
    print("PREPROCESSING: the incidence structure and its build time")
    print("=" * 78)
    print(f"{'T':>9} {'d':>3} {'q':>6} {'N0':>8} {'B':>8} {'deg':>6} "
          f"{'I_prof':>12} {'I_tok':>13} {'gain':>7} {'build ms':>9}")
    for r in sorted(rows, key=lambda r: (r["d"], r["T"])):
        print(f"{r['T']:>9,} {r['d']:>3} {r['q']:>6} {r['N0']:>8,} {r['B']:>8,} "
              f"{r['deg']:>6} {r['I_prof']:>12,} {r['I_tok']:>13,} "
              f"{r['pool_gain']:6.1f}x {r['build_ms']:>9.1f}")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--T", type=int, nargs="*", default=None)
    ap.add_argument("--d", type=int, nargs="*", default=None)
    ap.add_argument("--nb", type=int, default=8, help="batch * heads")
    ap.add_argument("--r", type=int, default=64, help="kernel feature rank r")
    ap.add_argument("--m", dest="r", type=int, help="alias for --r")
    ap.add_argument("--d-qk", dest="d_qk", type=int, default=64)
    ap.add_argument("--d-v", dest="d_v", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=128, help="scan chunk width c")
    ap.add_argument("--n-P", dest="n_P", type=int, default=0,
                    help="optional global channel |P| (Remark 3.4); 0 is the "
                         "main construction")
    ap.add_argument("--direct", action="store_true", default=True,
                    help="also time the unpooled long-range backend")
    ap.add_argument("--no-direct", dest="direct", action="store_false")
    ap.add_argument("--subtractive", action="store_true", default=True)
    ap.add_argument("--no-subtractive", dest="subtractive", action="store_false")
    ap.add_argument("--max-I-prof", dest="max_I_prof", type=int, default=40_000_000)
    ap.add_argument("--max-I-prof-torch", dest="max_I_prof_torch", type=int,
                    default=600_000,
                    help="skip the torch-path comparison above this nnz(C)")
    ap.add_argument("--max-I-tok", dest="max_I_tok", type=int, default=40_000_000)
    ap.add_argument("--baseline-max-T", dest="baseline_max_T", type=int, default=None)
    ap.add_argument("--dense-max-T", dest="dense_max_T", type=int, default=4096)
    ap.add_argument("--numeric-max-T", dest="numeric_max_T", type=int, default=16384)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--fit-min-T", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--tf32", action="store_true", default=True)
    ap.add_argument("--no-tf32", dest="tf32", action="store_false")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.T = args.T or [1024, 2048, 4096, 8192]
        args.d = args.d or [1, 2, 3]
        args.nb, args.r, args.d_qk, args.d_v = 2, 32, 32, 32
        args.repeat, args.warmup, args.decode_steps = 5, 2, 32
    if args.T is None:
        args.T = [2 ** k for k in range(12, 19)]
    if args.d is None:
        args.d = [1, 2, 3]
    if args.baseline_max_T is None:
        args.baseline_max_T = 2 ** 18 if args.device == "cuda" else 8192

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    if args.threads:
        torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
    else:
        print("NOTE: cpu timings are a shakeout only; they recover the exponent "
              "at best and say nothing about the tensor-core constant.")
        check_cpu_health()

    info = {"device": str(device), "dtype": args.dtype, "torch": torch.__version__,
            "python": platform.python_version(), "host": platform.node(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "triton": have_triton(), "tf32": args.tf32, "args": vars(args)}
    print(json.dumps({k: v for k, v in info.items() if k != "args"}, indent=2))
    print(f"\nsweep: T={args.T}  d={args.d}  nb={args.nb} r={args.r} "
          f"d_qk={args.d_qk} d_v={args.d_v} c={args.chunk} |P|={args.n_P}\n")

    rows: List[Dict] = []
    for d in args.d:
        print(f"d = {d}")
        for T in args.T:
            try:
                r = run_config(T, d, args, device, dtype)
            except (MemoryError, torch.cuda.OutOfMemoryError) as e:
                print(f"  T={T:>7} d={d}: out of memory ({e})")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            except (ValueError, NotImplementedError) as e:
                print(f"  T={T:>7} d={d}: skipped ({e})")
                continue
            if r:
                rows.append(r)
                write_rows(args.out, rows, info)
        print()

    if not rows:
        print("no configurations ran")
        return 1

    write_rows(args.out, rows, info)
    report(rows, args)
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

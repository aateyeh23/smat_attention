"""End-to-end cost of the d-Subset Routing models: training step, prefill, decode.

    python tasks/routing_bench.py --mode train   --T 1024 4096 16384 65536 --out results/routing_bench/train.jsonl
    python tasks/routing_bench.py --mode prefill --T 1024 4096 16384 65536 262144 --out ...
    python tasks/routing_bench.py --mode decode  --T 1024 4096 16384 65536 262144 --out ...

The models are ``tasks/routing_train.Router`` exactly as the routing table
trained them -- one layer, d_model 64, d_qk 32, d_v 32, feature rank 32, fp32,
batch 64 -- with SMat on the torch backends it trains with (its Triton kernels
are forward-only), the recurrent baselines on the chunked forms in
``src/smat/mixers/recurrent.py``, and softmax on PyTorch SDPA, which in fp32 is
the memory-efficient kernel rather than flash.  Only the hard-routing mask is
measured here; the learned-routing recall and PG-19 models are timed by
``tasks/pg19``'s benchmarks.

Modes.
  train    forward, MSE, backward, clip, Adam -- ``train_one``'s loop without
           its per-step numpy batch generation, which is data preparation
  prefill  inference forward over the whole window; SMat also on the Triton
           backend (the Figure 3 kernels) as ``smat-triton``
  decode   per-token latency over the recent block after the distant block is
           consumed: ``SmatDecoder`` for SMat, a preallocated KV cache for
           softmax, one-step recurrences for mamba2 / deltanet / gated_deltanet.
           Each decoder is checked against the full-sequence forward before it
           is timed.  loglinear has no step form here and is not decoded.

Inputs are random: the cost of a step depends on shapes, not on which
positions carry payloads.  One JSON row per (arm, T, batch).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / 'src', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from smat.mask import build_mask, d_max_geometric                     # noqa: E402
from smat.attention import featurise, smat_attention, to_device, SmatDecoder  # noqa: E402
from smat.mixers.recurrent import mamba2_loop, delta_loop, gated_delta_loop  # noqa: E402
from routing_ceiling import row_support_size                           # noqa: E402
from routing_train import Router                                       # noqa: E402

BASELINES = ('softmax', 'mamba2', 'deltanet', 'gated_deltanet', 'loglinear')
K = 4                                   # marked positions; cost does not depend on it


def arms_for(T, chunk, ds):
    dmax = d_max_geometric(T, chunk=chunk)
    return [('smat', d) for d in ds if d <= dmax] + [(a, 0) for a in BASELINES]


def make(arm, d, T, nb, chunk, device):
    spec = to_device(build_mask(T, d, chunk=chunk), device) if arm == 'smat' else None
    torch.manual_seed(0)
    model = Router(T, K, arm=arm, spec=spec, device=device, seed=0).to(device)
    nu = (torch.as_tensor(row_support_size(spec), device=device, dtype=torch.float32)
          if spec is not None else torch.ones(T, device=device))
    lo = spec.n if spec is not None else 0
    g = torch.Generator(device=device).manual_seed(0)
    pay = torch.randn(nb, T, K, device=device, generator=g)
    rows = torch.randint(lo, T, (nb,), device=device, generator=g)
    bits = torch.randint(0, 2, (nb, K), device=device, generator=g).float()
    y = torch.randn(nb, K, device=device, generator=g)
    return model, spec, nu, (pay, rows, bits, y)


def cuda_times(fn, warmup, steps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = []
    for _ in range(steps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return dict(median_ms=float(np.median(out)), p25_ms=float(np.percentile(out, 25)),
                p75_ms=float(np.percentile(out, 75)), n=len(out),
                peak_gb=torch.cuda.max_memory_allocated() / 1e9)


# ------------------------------------------------------------------ train / prefill
def run_train(model, nu, batch, args):
    pay, rows, bits, y = batch
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def step():
        loss = F.mse_loss(model(pay, rows, bits, nu), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return cuda_times(step, args.warmup, args.steps)


def smat_forward_triton(model, pay, rows, nu):
    """Router.forward for SMat with the Triton backends (forward only)."""
    nb, T, _ = pay.shape
    X = model.pos(torch.arange(T, device=pay.device)).unsqueeze(0) + model.pay_in(pay)
    Phi, Psi, Vb = featurise(model.Wq(X), model.Wk(X), model.Wv(X), model.phi)
    o = smat_attention(Phi, Psi, Vb, model.spec, chunk=128, acc_dtype=torch.float32,
                       scan_backend='auto', incidence_backend='auto') * nu.view(1, T, 1)
    return model.Wo(o[torch.arange(nb, device=pay.device), rows])


@torch.no_grad()
def run_prefill(model, nu, batch, args, triton=False):
    pay, rows, bits, _ = batch
    model.eval()
    fn = ((lambda: smat_forward_triton(model, pay, rows, nu)) if triton
          else (lambda: model(pay, rows, bits, nu)))
    return cuda_times(fn, args.warmup, args.steps)


# ------------------------------------------------------------------ decode
class Stepper:
    """One token of Router's mixer at a time, for every arm with a step form."""

    def __init__(self, model, spec, nu, pay, n):
        self.m, self.spec, self.nu, self.pay, self.t = model, spec, nu, pay, n
        nb, T, _ = pay.shape
        dev = pay.device
        dq, dv = model.Wq.out_features, model.Wv.out_features
        if model.arm == 'smat':
            X = model.pos(torch.arange(n, device=dev)).unsqueeze(0) + model.pay_in(pay[:, :n])
            _, Psi, Vb = featurise(model.Wq(X), model.Wk(X), model.Wv(X), model.phi)
            self.dec = SmatDecoder(Psi, Vb, spec, acc_dtype=torch.float32)
            self.cache = dict(U=self.dec.U, S=self.dec.S)
        elif model.arm == 'softmax':
            X = model.pos(torch.arange(n, device=dev)).unsqueeze(0) + model.pay_in(pay[:, :n])
            self.K = torch.empty(nb, 1, T, dq, device=dev)
            self.V = torch.empty(nb, 1, T, dv, device=dev)
            self.K[:, 0, :n], self.V[:, 0, :n] = model.Wk(X), model.Wv(X)
            self.cache = dict(K=self.K[:, :, :T], V=self.V[:, :, :T])
        else:
            # the state's value does not change the cost of a step; its update
            # rule is checked against the sequential reference in check_stepper
            self.S = torch.zeros(nb, dq, dv, device=dev)
            self.cache = dict(S=self.S)

    def step(self):
        m, t = self.m, self.t
        x = m.pos.weight[t].unsqueeze(0) + m.pay_in(self.pay[:, t])      # (nb, d_model)
        q, k, v = m.Wq(x), m.Wk(x), m.Wv(x)
        if m.arm == 'smat':
            Phi, Psi, Vb = featurise(q, k, v, m.phi)
            o = self.dec.step(Phi, Psi, Vb) * self.nu[t]
        elif m.arm == 'softmax':
            self.K[:, 0, t], self.V[:, 0, t] = k, v
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                o = F.scaled_dot_product_attention(q[:, None, None], self.K[:, :, :t + 1],
                                                   self.V[:, :, :t + 1])[:, 0, 0]
        else:
            o = self._recurrent(x, q, k, v)
        self.t += 1
        return m.Wo(o)

    def _recurrent(self, x, q, k, v):
        m, S = self.m, self.S
        if m.arm == 'mamba2':
            logA, dt = m.gate(x)
            S.mul_(logA.exp()[:, None, None]).add_(dt[:, None, None] * (k.unsqueeze(-1) @ v.unsqueeze(-2)))
        else:
            k = F.normalize(k, dim=-1)
            if m.arm == 'deltanet':
                b = m.gate(x)
            else:
                logA, b = m.gate(x)
                S.mul_(logA.exp()[:, None, None])
            delta = b[:, None] * (v - (k.unsqueeze(-2) @ S).squeeze(-2))
            S.add_(k.unsqueeze(-1) @ delta.unsqueeze(-2))
        return (q.unsqueeze(-2) @ S).squeeze(-2)


@torch.no_grad()
def check_stepper(model, spec, nu, pay):
    """Decoder output against the full-sequence computation it replaces."""
    m, nb, T = model, pay.shape[0], pay.shape[1]
    X = m.pos(torch.arange(T, device=pay.device)).unsqueeze(0) + m.pay_in(pay)
    Q, Kk, V = m.Wq(X), m.Wk(X), m.Wv(X)
    if m.arm == 'smat':
        n, steps = spec.n, min(64, T - spec.n)
        Phi, Psi, Vb = featurise(Q, Kk, V, m.phi)
        ref = smat_attention(Phi, Psi, Vb, spec, chunk=128, acc_dtype=torch.float32,
                             scan_backend='torch', incidence_backend='torch') * nu.view(1, T, 1)
        ref = ref[:, n:n + steps]
    elif m.arm == 'softmax':
        n, steps = T // 2, min(64, T // 2)
        ref = F.scaled_dot_product_attention(Q[:, None], Kk[:, None], V[:, None],
                                             is_causal=True)[:, 0, n:n + steps]
    else:                                   # from an empty state, against the loops
        n, steps = 0, min(64, T)
        sl = slice(0, steps)
        if m.arm == 'mamba2':
            logA, dt = m.gate(X[:, sl]); ref = mamba2_loop(Q[:, sl], Kk[:, sl], V[:, sl], logA, dt)
        elif m.arm == 'deltanet':
            ref = delta_loop(Q[:, sl], Kk[:, sl], V[:, sl], m.gate(X[:, sl]))
        else:
            logA, b = m.gate(X[:, sl]); ref = gated_delta_loop(Q[:, sl], Kk[:, sl], V[:, sl], logA, b)
    st = Stepper(m, spec, nu, pay, n)
    got = []
    for _ in range(steps):
        o = st.step()
        got.append(o)
    # compare pre-readout outputs: undo Wo by comparing through it
    ref_out = m.Wo(ref)
    got = torch.stack(got, 1)
    return float(((got - ref_out).norm() / ref_out.norm().clamp_min(1e-30)).item())


@torch.no_grad()
def run_decode(model, spec, nu, batch, args):
    pay = batch[0]
    model.eval()
    T = pay.shape[1]
    rel = check_stepper(model, spec, nu, pay)
    n = spec.n if spec is not None else T // 2
    steps = min(args.decode_steps, T - n - 16)
    st = Stepper(model, spec, nu, pay, n)
    cache = sum(v.numel() * v.element_size() for v in st.cache.values())
    res = cuda_times(st.step, 8, steps)
    return dict(**res, prompt_tokens=n, cache_mb=cache / 1e6, decode_rel_err=rel,
                tokens_per_second=pay.shape[0] / (res['median_ms'] / 1e3))


# ------------------------------------------------------------------ driver
def provenance():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    return dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                capability='sm_%d%d' % torch.cuda.get_device_capability(), git_commit=commit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('train', 'prefill', 'decode'), required=True)
    ap.add_argument('--T', type=int, nargs='+', default=[1024, 4096, 16384, 65536])
    ap.add_argument('--d', type=int, nargs='+', default=[1, 2, 3, 4])
    ap.add_argument('--arms', nargs='+', default=None, help='restrict to these arm names')
    ap.add_argument('--nb', type=int, nargs='+', default=[64])
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--decode-steps', type=int, default=64)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False     # the table trained in plain fp32
    prov = provenance()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for T in args.T:
        for arm, d in arms_for(T, args.chunk, args.d):
            if args.arms and arm not in args.arms:
                continue
            if args.mode == 'decode' and arm == 'loglinear':
                continue
            if arm == 'loglinear' and T > 16384:
                # fenwick_levels materialises a (T, T) int64 table in a python loop
                row = dict(arm=arm, d=0, T=T, nb=args.nb[0], mode=args.mode, fp='fp32', **prov,
                           status='skipped', error='dense (T, T) level table: %.0f GB' % (T * T * 8 / 1e9))
                with open(args.out, 'a') as fh:
                    fh.write(json.dumps(row) + '\n')
                print('ROW ' + json.dumps(row), flush=True)
                continue
            variants = [False, True] if (args.mode == 'prefill' and arm == 'smat') else [False]
            for nb in args.nb:
                for tri in variants:
                    name = 'smat-triton' if tri else arm
                    row = dict(arm=name, d=d, T=T, nb=nb, mode=args.mode, fp='fp32', **prov)
                    t0 = time.perf_counter()
                    try:
                        model, spec, nu, batch = make(arm, d, T, nb, args.chunk, dev)
                        row['params'] = sum(p.numel() for p in model.parameters())
                        if spec is not None:
                            row.update(q=int(spec.q), B=int(spec.B), N0=int(spec.N0), n=int(spec.n))
                        if args.mode == 'train':
                            row.update(run_train(model, nu, batch, args))
                        elif args.mode == 'prefill':
                            row.update(run_prefill(model, nu, batch, args, triton=tri))
                        else:
                            row.update(run_decode(model, spec, nu, batch, args))
                        row['status'] = 'ok'
                    except torch.OutOfMemoryError as e:
                        row.update(status='oom', error=str(e).splitlines()[0][:200])
                    except Exception as e:
                        row.update(status='error', error=f'{type(e).__name__}: {e}'[:400])
                    row['wall_seconds'] = time.perf_counter() - t0
                    model = spec = nu = batch = None       # free before the next arm
                    torch.cuda.empty_cache()
                    with open(args.out, 'a') as fh:
                        fh.write(json.dumps(row) + '\n')
                    print('ROW ' + json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

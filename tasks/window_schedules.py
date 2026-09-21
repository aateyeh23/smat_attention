"""Where should the distant/recent boundary sit when the final length is unknown?

Compares boundary schedules for a content-addressed SMat layer on top of a
Mamba-2-style decayed linear-attention recurrence, trained at ``T_train`` and
evaluated at 1x .. 8x that length.

Schedules (each row t reads distant keys j < n(t) through G, and keys in
[n(t), t] through the recurrence, which is reset at n(t)):

  none         n(t) = 0                          pure recurrence (d = 1)
  half         n(t) = T_train/2 for t >= T_train/2
               the draft's mask at T_train; at longer lengths this is what the
               fixed-horizon decoder does when it keeps its prefill boundary
  half_oracle  n(t) = T/2 for t >= T/2           the draft's mask rebuilt at the
               evaluation length (needs T in advance; evaluation only)
  win4, win8   stepped window, blocks of m = T_train/k:
               n(t) = (b-1) m for block b >= 2, else 0
  dbl          doubling: n(t) = 2^(floor(log2 t) - 1) for t >= T_train/2

Every schedule is a list of row segments (lo, hi, n).  Consecutive segments
alternate between two recurrent runs, and a run is reset at the boundaries of
its own segments, so each row reads a recurrence started exactly at n(t).  G
pools the distant prefix [0, n) once per distinct n, incrementally.

Tasks:
  mqar   placed-pair multi-query associative recall; training places pairs and
         queries uniformly, evaluation places a pair region at a controlled
         distance D before a query region at the end of the sequence.
  pg19   byte-level PG-19; evaluation reports loss by position on long windows.

Hash: fixed random n-gram hash of token ids, one table per head (no learning).
Key j is filed under the hash of the g tokens before it; query t looks up the
hash of the g tokens ending at t (g = 1 for MQAR, 3 for PG-19).

  python tasks/window_schedules.py selftest
  python tasks/window_schedules.py mqar --sched win4 --seed 0 --steps 6000
  python tasks/window_schedules.py pg19 --sched win4 --minutes 35
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from smat.mask import build_mask            # noqa: E402

SCHEDULES = ("none", "half", "half_oracle", "win4", "win8", "dbl")


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------
def segments(sched: str, T: int, T_train: int):
    """Row segments (lo, hi, n) covering [0, T) in order."""
    if sched == "none":
        return [(0, T, 0)]
    if sched in ("half", "half_oracle"):
        n = T_train // 2 if sched == "half" else T // 2
        segs = [(0, min(n, T), 0)]
        if n < T:
            segs.append((n, T, n))
        return segs
    if sched.startswith("win"):
        m = T_train // int(sched[3:])
        segs = [(0, min(2 * m, T), 0)]
        b = 2
        while b * m < T:
            segs.append((b * m, min((b + 1) * m, T), (b - 1) * m))
            b += 1
        return segs
    if sched == "dbl":
        h = T_train // 2
        segs = [(0, min(h, T), 0)]
        lo = h
        while lo < T:
            segs.append((lo, min(2 * lo, T), lo // 2))
            lo *= 2
        return segs
    raise ValueError(sched)


def runs_of(segs):
    """Assign segment i to run i % 2 and return each run's reset positions."""
    resets = ({0}, {0})
    for i, (_, _, n) in enumerate(segs):
        resets[i % 2].add(n)
    for i, (lo, hi, n) in enumerate(segs):       # a run must not reset inside its own segment's window
        later = [s[2] for s in segs[i + 2::2]]
        assert all(r >= hi for r in later), (segs, i)
    return resets


def n_of_row(segs, T):
    n = np.zeros(T, dtype=np.int64)
    for lo, hi, b in segs:
        n[lo:hi] = b
    return n


# ---------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------
def decayed_scan(Phi, Psi, Vb, logA, resets, C):
    """S_t = a_t S_{t-1} + psi_t vb_t^T, y_t = phi_t^T S_t, with S zeroed before
    every position in ``resets`` (multiples of C).  (B,T,r),(B,T,r),(B,T,p),(B,T)."""
    B, T, r = Phi.shape
    p = Vb.shape[-1]
    pad = (-T) % C
    if pad:
        Phi = F.pad(Phi, (0, 0, 0, pad)); Psi = F.pad(Psi, (0, 0, 0, pad))
        Vb = F.pad(Vb, (0, 0, 0, pad)); logA = F.pad(logA, (0, pad))
    nc = (T + pad) // C
    Ph = Phi.reshape(B, nc, C, r); Ps = Psi.reshape(B, nc, C, r)
    Vc = Vb.reshape(B, nc, C, p); L = torch.cumsum(logA.reshape(B, nc, C), -1)
    tri = torch.ones(C, C, device=Phi.device, dtype=torch.bool).tril()
    W = torch.exp((L.unsqueeze(-1) - L.unsqueeze(-2)).masked_fill(~tri, float("-inf")))
    out = ((Ph @ Ps.transpose(-1, -2)) * W) @ Vc
    D = (Ps * torch.exp(L[..., -1:] - L).unsqueeze(-1)).transpose(-1, -2) @ Vc
    aC = torch.exp(L[..., -1])
    S = Phi.new_zeros(B, r, p)
    carry = []
    for b in range(nc):
        if b * C in resets:
            S = torch.zeros_like(S)
        carry.append((Ph[:, b] * torch.exp(L[:, b]).unsqueeze(-1)) @ S)
        S = aC[:, b].view(B, 1, 1) * S + D[:, b]
    out = out + torch.stack(carry, 1)
    return out.reshape(B, nc * C, p)[:, :T]


def g_read(Phi, PsiG, Vb, ck, cq, segs, N0):
    """Augmented long-range term.  Row t in segment (lo, hi, n) reads
    sum_{j<n, ck_j = cq_t} (phi_t . psiG_j) vb_j.  Pools are incremental in n."""
    B, T, r = Phi.shape
    p = Vb.shape[-1]
    bounds = sorted({n for _, _, n in segs if n > 0})
    tables, acc, prev = {}, None, 0
    for n in bounds:
        cells = ck[:, prev:n]
        Pc = Phi.new_zeros(B, n - prev, N0, r).scatter(
            2, cells[..., None, None].expand(-1, -1, 1, r), PsiG[:, prev:n, None, :])
        inc = Pc.flatten(2).transpose(1, 2) @ Vb[:, prev:n]          # (B, N0*r, p)
        acc = inc if acc is None else acc + inc
        tables[n], prev = acc, n
    pieces = []
    for lo, hi, n in segs:
        if n == 0:
            pieces.append(Phi.new_zeros(B, hi - lo, p))
            continue
        Qc = Phi.new_zeros(B, hi - lo, N0, r).scatter(
            2, cq[:, lo:hi, None, None].expand(-1, -1, 1, r), Phi[:, lo:hi, None, :])
        pieces.append(Qc.flatten(2) @ tables[n])
    return torch.cat(pieces, 1)


class Mixer(nn.Module):
    def __init__(self, d_model, h, r, N0, vocab, gram, use_g, seed, dt=(0.1, 1.0), A=(1.0, 16.0)):
        super().__init__()
        self.h, self.dh, self.N0, self.gram, self.use_g = h, d_model // h, N0, gram, use_g
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.qn, self.kn = nn.LayerNorm(self.dh), nn.LayerNorm(self.dh)
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("Wphi", torch.randn(r, self.dh, generator=gen) / math.sqrt(self.dh))
        self.w_dt = nn.Linear(d_model, h)
        self.A_log = nn.Parameter(torch.log(torch.empty(h).uniform_(*A)))
        self.dt_range = dt
        if use_g:
            self.w_g = nn.Linear(d_model, h)
            gen = torch.Generator().manual_seed(1000 + seed)
            self.register_buffer("hash", torch.randint(0, N0, (gram, vocab, h), generator=gen))

    def phi(self, x):
        """Fixed random elu+1 features, the repository's default positive map."""
        return F.elu(x @ self.Wphi.T) + 1.0

    def reset_gates(self):
        with torch.no_grad():
            nn.init.zeros_(self.w_dt.weight)
            dt0 = torch.exp(torch.empty(self.h).uniform_(*map(math.log, self.dt_range)))
            self.w_dt.bias.copy_(dt0 + torch.log(-torch.expm1(-dt0)))
            if self.use_g:
                nn.init.zeros_(self.w_g.weight); nn.init.zeros_(self.w_g.bias)

    def cells(self, ids):
        """(nb, T) -> key cells, query cells, each (nb*h, T)."""
        nb, T = ids.shape
        g = self.gram
        pad = F.pad(ids, (g, 0))                                      # (nb, T+g), BOS id 0 in front
        ck = sum(self.hash[i][pad[:, i:i + T]] for i in range(g)) % self.N0        # tokens t-g .. t-1
        cq = sum(self.hash[i][pad[:, i + 1:i + 1 + T]] for i in range(g)) % self.N0  # tokens t-g+1 .. t
        f = lambda c: c.permute(0, 2, 1).reshape(nb * self.h, T)
        return f(ck), f(cq)

    def forward(self, x, ids, segs):
        nb, T, C = x.shape
        q, k, v = self.qkv(x).view(nb, T, 3, self.h, self.dh).unbind(2)
        q, k = self.qn(q), self.kn(k)
        kd = self.Wphi.dtype                                           # fp32 in training, fp64 in the self-test
        with torch.autocast(device_type=x.device.type, enabled=False):
            fold = lambda t: t.transpose(1, 2).reshape(nb * self.h, T, -1).to(kd)
            Phi, Psi, V = self.phi(fold(q)), self.phi(fold(k)), fold(v)
            Vb = torch.cat([V, torch.ones_like(V[..., :1])], -1)
            xf = x.to(kd)
            dt = F.softplus(self.w_dt(xf))                             # (nb, T, h)
            logA = fold(-(dt * torch.exp(self.A_log.to(kd))).unsqueeze(-1)).squeeze(-1)
            Psi_r = Psi * fold(dt.unsqueeze(-1))
            resets = runs_of(segs)
            H = None
            for r_id in (0, 1):
                rows = [s for i, s in enumerate(segs) if i % 2 == r_id]
                if not rows:
                    continue
                Hr = decayed_scan(Phi, Psi_r, Vb, logA, resets[r_id], self.chunk)
                mask = torch.zeros(T, device=x.device, dtype=Hr.dtype)
                for lo, hi, _ in rows:
                    mask[lo:hi] = 1
                H = Hr * mask.view(1, T, 1) if H is None else H + Hr * mask.view(1, T, 1)
            if self.use_g and any(n > 0 for _, _, n in segs):
                ck, cq = self.cells(ids)
                g = fold(torch.sigmoid(self.w_g(xf)).unsqueeze(-1))
                H = H + g_read(Phi, Psi * g, Vb, ck, cq, segs, self.N0)
            o = H[..., :-1] / H[..., -1:].clamp_min(1e-6)
            o = o.view(nb, self.h, T, self.dh).transpose(1, 2).reshape(nb, T, C)
        return self.out(o.to(x.dtype))


class Block(nn.Module):
    def __init__(self, d_model, **kw):
        super().__init__()
        self.ln0, self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, 3, groups=d_model, padding=2)
        self.mix = Mixer(d_model, **kw)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, ids, segs):
        T = x.shape[1]
        x = x + self.conv(self.ln0(x).transpose(1, 2))[..., :T].transpose(1, 2)
        x = x + self.mix(self.ln1(x), ids, segs)
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    """No positional embedding: order comes from the convolution and the recurrence,
    so evaluation past T_train does not depend on an extrapolated position table."""

    def __init__(self, vocab, d_model, h, layers, r, N0, gram, sched, T_train, chunk, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.sched, self.T_train = sched, T_train
        use_g = sched != "none"
        self.tok = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model, h=h, r=r, N0=N0, vocab=vocab, gram=gram,
                                           use_g=use_g, seed=seed + 7 * i) for i in range(layers)])
        for b in self.blocks:
            b.mix.chunk = chunk
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init)
        for b in self.blocks:
            b.mix.reset_gates()

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, sched=None):
        T = ids.shape[1]
        segs = segments(sched or self.sched, T, self.T_train)
        x = self.tok(ids)
        for b in self.blocks:
            x = b(x, ids, segs)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
class MQAR:
    NOISE, K, V = 64, 256, 256

    def __init__(self, P, seed):
        self.P, self.rng = P, np.random.default_rng(seed)
        self.vocab = 1 + self.NOISE + self.K + self.V

    def _fill(self, x, y, pair_slots, query_slots):
        P, rng = self.P, self.rng
        keys = 1 + self.NOISE + rng.choice(self.K, P, replace=False)
        vals = 1 + self.NOISE + self.K + rng.integers(0, self.V, P)
        for s, i in zip(pair_slots, range(P)):
            x[s], x[s + 1] = keys[i], vals[i]
        order = rng.permutation(P)
        for s, i in zip(query_slots, order):
            x[s], x[s + 1] = keys[i], vals[i]
            y[s] = vals[i]

    def batch(self, nb, T, layout="uniform", D=0, W=256):
        rng, P = self.rng, self.P
        x = rng.integers(1, 1 + self.NOISE, (nb, T))
        x[:, 0] = 0
        y = np.full((nb, T), -100)
        for b in range(nb):
            if layout == "uniform":
                slots = np.sort(rng.choice((T - 2) // 2, 2 * P, replace=False)) * 2 + 2
                lab = rng.permutation(np.repeat(np.arange(P), 2))
                first = {}
                pair, query = [None] * P, [None] * P
                for s, l in zip(slots, lab):
                    if l in first:
                        query[l] = s
                    else:
                        first[l] = s
                        pair[l] = s
                keys = 1 + self.NOISE + rng.choice(self.K, P, replace=False)
                vals = 1 + self.NOISE + self.K + rng.integers(0, self.V, P)
                for i in range(P):
                    x[b, pair[i]], x[b, pair[i] + 1] = keys[i], vals[i]
                    x[b, query[i]], x[b, query[i] + 1] = keys[i], vals[i]
                    y[b, query[i]] = vals[i]
            else:
                a = T - 2 * W - D
                assert a >= 2, (T, D)
                ps = np.sort(rng.choice(W // 2, P, replace=False)) * 2 + a
                qs = np.sort(rng.choice(W // 2 - 1, P, replace=False)) * 2 + (T - W)
                self._fill(x[b], y[b], ps, qs)
        return torch.as_tensor(x), torch.as_tensor(y)


class PG19Bytes:
    def __init__(self, root, seed):
        self.train = np.memmap(os.path.join(root, "train.bin"), dtype=np.uint8, mode="r")
        self.test = np.memmap(os.path.join(root, "test.bin"), dtype=np.uint8, mode="r")
        self.test_off = np.load(os.path.join(root, "test_offsets.npy"))
        self.rng = np.random.default_rng(seed)
        self.vocab = 256

    def batch(self, nb, T):
        ix = self.rng.integers(0, len(self.train) - T - 1, nb)
        a = np.stack([self.train[i:i + T + 1] for i in ix]).astype(np.int64)
        return torch.as_tensor(a[:, :-1]), torch.as_tensor(a[:, 1:])

    def eval_windows(self, n, T, seed=123):
        """n windows of T+1 bytes, each inside one test book."""
        rng = np.random.default_rng(seed)
        lo, hi = self.test_off[:-1], self.test_off[1:]
        ok = np.nonzero(hi - lo > T + 1)[0]
        out = []
        for _ in range(n):
            bk = rng.choice(ok)
            s = rng.integers(lo[bk], hi[bk] - T - 1)
            out.append(self.test[s:s + T + 1])
        a = np.stack(out).astype(np.int64)
        return torch.as_tensor(a[:, :-1]), torch.as_tensor(a[:, 1:])


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------
def geometry(T_train, d):
    spec = build_mask(T_train, d)
    return spec.q, spec.N0


def make_opt(model, lr, wd, steps, warm=200):
    gate = [p for n, p in model.named_parameters() if any(k in n for k in ("w_dt", "A_log", "w_g"))]
    gid = {id(p) for p in gate}
    rest = [p for p in model.parameters() if id(p) not in gid]
    opt = torch.optim.AdamW([{"params": rest, "weight_decay": wd}, {"params": gate, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    sched = lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))
    return opt, sched


def train_loop(model, get_batch, args, dev, log):
    opt, sched = make_opt(model, args.lr, args.wd, args.steps)
    model.train()
    t0 = time.time()
    step = 0
    budget = args.minutes * 60 if args.minutes else None
    while step < args.steps:
        if budget and time.time() - t0 > budget:
            log(f"time budget reached at step {step}")
            break
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step)
        x, y = get_batch()
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            log(f"step {step} loss {loss.item():.4f} {1000 * (time.time() - t0) / step:.0f} ms/step")
    return step, time.time() - t0


def main_mqar(args, dev):
    q, N0 = geometry(args.T, args.d)
    task = MQAR(args.pairs, args.seed)
    model = Model(task.vocab, args.d_model, args.heads, args.layers, args.r, N0, 1,
                  args.sched, args.T, args.chunk, args.seed).to(dev)
    tag = f"mqar {args.sched} s{args.seed}"
    log = lambda s: print(f"[{tag}] {s}", flush=True)
    log(f"q={q} N0={N0} params={sum(p.numel() for p in model.parameters())}")
    steps, secs = train_loop(model, lambda: task.batch(args.nb, args.T), args, dev, log)
    model.eval()
    rows = []
    eval_task = MQAR(args.pairs, 10_000 + args.seed)
    scheds = [args.sched] + (["half_oracle"] if args.sched == "half" else [])
    for Te in args.eval_T:
        Ds = [D for D in args.D if D <= Te - 2 * 256 - 2] + [Te - 2 * 256 - 2]
        layouts = [("uniform", 0)] + [("placed", D) for D in sorted(set(Ds))]
        nb = max(1, args.eval_tokens // Te)
        for lay, D in layouts:
            for sc in scheds:
                ok = tot = 0
                for it in range(args.eval_batches):
                    x, y = eval_task.batch(nb, Te, lay, D)
                    x, y = x.to(dev), y.to(dev)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                        pred = model(x, sched=sc).argmax(-1)
                    m = y != -100
                    ok += (pred[m] == y[m]).sum().item(); tot += int(m.sum())
                acc = ok / max(tot, 1)
                # where the pair region sits relative to each schedule's boundary at the query rows
                segs = segments(sc, Te, args.T)
                nq = n_of_row(segs, Te)[Te - 256:]
                pair_end = Te - 256 - D
                frac_g = float(np.mean(pair_end <= nq)) if lay == "placed" else float("nan")
                row = dict(task="mqar", sched=sc, trained=args.sched, seed=args.seed, d=args.d, q=q, N0=N0,
                           T_train=args.T, T_eval=Te, layout=lay, D=D, acc=acc, n_queries=tot,
                           pairs_in_G=frac_g, steps=steps, train_sec=round(secs))
                rows.append(row)
                log(f"eval T={Te} {lay} D={D} [{sc}] acc {acc:.4f}  pairs-in-G {frac_g:.2f}")
    write_rows(args.out, rows)


def main_pg19(args, dev):
    q, N0 = geometry(args.T, args.d)
    data = PG19Bytes(args.data, args.seed)
    model = Model(256, args.d_model, args.heads, args.layers, args.r, N0, 3,
                  args.sched, args.T, args.chunk, args.seed).to(dev)
    tag = f"pg19 {args.sched} s{args.seed}"
    log = lambda s: print(f"[{tag}] {s}", flush=True)
    log(f"q={q} N0={N0} params={sum(p.numel() for p in model.parameters())}")
    steps, secs = train_loop(model, lambda: data.batch(args.nb, args.T), args, dev, log)
    model.eval()
    rows = []
    scheds = [args.sched] + (["half_oracle"] if args.sched == "half" else [])
    bin_w = args.bin
    with torch.no_grad():
        for Te in args.eval_T:
            nb = max(1, args.eval_tokens // Te)
            for sc in scheds:
                if sc != "half_oracle" and Te != max(args.eval_T):
                    continue          # prefix-consistent schedules: the longest window gives every prefix
                tot = torch.zeros(Te, dtype=torch.float64)
                cnt = 0
                for it in range(args.eval_batches):
                    x, y = data.eval_windows(nb, Te, seed=500 + it)
                    x, y = x.to(dev), y.to(dev)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                        logits = model(x, sched=sc)
                    nll = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")
                    tot += nll.sum(0).double().cpu(); cnt += nb
                per = (tot / cnt).numpy() / math.log(2)                # bits per byte
                for b0 in range(0, Te, bin_w):
                    rows.append(dict(task="pg19", sched=sc, trained=args.sched, seed=args.seed, d=args.d, q=q,
                                     N0=N0, T_train=args.T, T_eval=Te, pos_lo=b0, pos_hi=b0 + bin_w,
                                     bpb=float(per[b0:b0 + bin_w].mean()), n_windows=cnt,
                                     steps=steps, train_sec=round(secs)))
                summ = {f"[{a},{b})": round(float(per[a:b].mean()), 4)
                        for a, b in ((0, args.T // 2), (args.T // 2, args.T), (args.T, 2 * args.T),
                                     (2 * args.T, 4 * args.T), (4 * args.T, 8 * args.T)) if b <= Te}
                log(f"eval T={Te} [{sc}] bpb by range {summ}")
    write_rows(args.out, rows)


def write_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# self-test: fast path == dense masked reference, and prefix consistency
# ---------------------------------------------------------------------------
def selftest():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    T_train, chunk, vocab = 64, 4, 11
    worst = 0.0
    for sched in ("none", "half", "half_oracle", "win4", "win8", "dbl"):
        for T in (64, 128, 256):
            mx = Mixer(24, h=2, r=5, N0=7, vocab=vocab, gram=2, use_g=sched != "none", seed=1).double()
            mx.chunk = chunk
            with torch.no_grad():
                mx.w_dt.weight.normal_(0, 0.3); mx.A_log.normal_(0, 0.3)
                if mx.use_g:
                    mx.w_g.weight.normal_(0, 0.3)
            ids = torch.randint(0, vocab, (2, T))
            x = torch.randn(2, T, 24)
            segs = segments(sched, T, T_train)
            fast = mx(x, ids, segs)
            # dense reference
            nb, h, dh = 2, 2, 12
            q, k, v = mx.qkv(x).view(nb, T, 3, h, dh).unbind(2)
            q, k = mx.qn(q), mx.kn(k)
            fold = lambda t: t.transpose(1, 2).reshape(nb * h, T, -1)
            Phi, Psi, V = mx.phi(fold(q)), mx.phi(fold(k)), fold(v)
            Vb = torch.cat([V, torch.ones_like(V[..., :1])], -1)
            dt = F.softplus(mx.w_dt(x))
            lam = torch.cumsum(fold(-(dt * torch.exp(mx.A_log)).unsqueeze(-1)).squeeze(-1), -1)
            dtf = fold(dt.unsqueeze(-1)).squeeze(-1)
            nrow = torch.as_tensor(n_of_row(segs, T))
            tt = torch.arange(T)
            rec = (tt[None, :] >= nrow[:, None]) & (tt[None, :] <= tt[:, None])
            Wrec = rec.double() * torch.exp(lam[:, :, None] - lam[:, None, :]) * dtf[:, None, :]
            A = (Phi @ Psi.transpose(1, 2)) * Wrec
            if mx.use_g:
                ck, cq = mx.cells(ids)
                gg = fold(torch.sigmoid(mx.w_g(x)).unsqueeze(-1)).squeeze(-1)
                dist = (tt[None, :] < nrow[:, None]).double()
                same = (cq[:, :, None] == ck[:, None, :]).double()
                A = A + (Phi @ Psi.transpose(1, 2)) * dist * same * gg[:, None, :]
            H = A @ Vb
            o = (H[..., :-1] / H[..., -1:]).view(nb, h, T, dh).transpose(1, 2).reshape(nb, T, 24)
            ref = mx.out(o)
            err = ((fast - ref).abs().max() / ref.abs().max()).item()
            worst = max(worst, err)
            print(f"selftest {sched:12s} T={T:4d}  rel.err {err:.2e}")
    # prefix consistency: rows of a length-T forward equal the first T rows of a longer forward
    m = Model(vocab, 24, 2, 2, 5, 7, 2, "win4", T_train, chunk, 0).double()
    ids = torch.randint(0, vocab, (2, 256))
    for sched in ("win4", "win8", "dbl", "half", "none", "half_oracle"):
        a = m(ids, sched=sched)
        b = m(ids[:, :128], sched=sched)
        d = (a[:, :128] - b).abs().max().item()
        print(f"prefix consistency {sched:12s} max diff {d:.2e}")
    assert worst < 1e-10, worst
    print(f"SELFTEST OK worst rel.err {worst:.2e}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["mqar", "pg19", "selftest"])
    ap.add_argument("--sched", choices=[s for s in SCHEDULES if s != "half_oracle"], default="win4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--nb", type=int, default=32)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--minutes", type=float, default=0.0, help="stop training after this many minutes")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--eval-T", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--D", type=int, nargs="+", default=[0, 256, 1024, 3072])
    ap.add_argument("--eval-tokens", type=int, default=32768)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--bin", type=int, default=128)
    # set SMAT_PG19_BYTES in experiments/site.conf; data/ is gitignored
    ap.add_argument("--data", default=os.environ.get("SMAT_PG19_BYTES", "data/pg19_bytes"))
    ap.add_argument("--out", default="results/window_schedules/mqar.csv")
    args = ap.parse_args(argv)
    if args.task == "selftest":
        return selftest()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({**vars(args), "device": dev,
                      "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else ""}), flush=True)
    (main_mqar if args.task == "mqar" else main_pg19)(args, dev)


if __name__ == "__main__":
    main()

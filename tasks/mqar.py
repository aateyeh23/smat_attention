"""Multi-Query Associative Recall (Arora et al., Zoology) through the LM stack.

Layout, following Zoology's generator: the first ``2 * n_pairs`` positions hold
key-value pairs ``k1 v1 k2 v2 ...`` with keys and values drawn without
replacement from disjoint vocabularies.  The remaining positions are noise
tokens except for queries: a key from the context, whose label is its value
(the model predicts the value from the hidden state at the key).  Queries are
placed at random with a gap of at least two, all in the recent segment.

Content-addressed by design: a query must find its key wherever it sits.  Under
a SMAT mask a recent row sees only the landmark columns on its hyperplane, so
most pairs are invisible to most queries; this is the limits experiment, not
the routing one.
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))          # the smat package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling task modules
sys.path.insert(0, HERE)
from smat.mask import build_mask                              # noqa: E402
from smat.attention import to_device                               # noqa: E402
from smat import model as train_lm                                               # noqa: E402


class MQAR:
    def __init__(self, T, n_pairs, n_queries, n_keys, device, seed=0, ctx_start=0, pad_noise=False):
        assert ctx_start + 2 * n_pairs <= T
        self.pad_noise = pad_noise      # a single PAD token instead of random noise tokens
        self.T, self.n_pairs, self.n_queries, self.n_keys = T, n_pairs, n_queries, n_keys
        self.device, self.ctx_start = device, ctx_start
        self.rng = np.random.default_rng(seed)
        self.KEY0, self.VAL0, self.NOISE0 = 0, n_keys, 2 * n_keys
        self.vocab = 3 * n_keys
        self.ctx_end = ctx_start + 2 * n_pairs
        self.q_lo = max(self.ctx_end, T // 2)      # queries in the recent half only

    def batch(self, nb):
        T, N = self.T, self.n_pairs
        x = (np.full((nb, T), self.NOISE0, dtype=np.int64) if self.pad_noise
             else self.rng.integers(self.NOISE0, self.vocab, size=(nb, T)))
        y = np.full((nb, T), -100, dtype=np.int64)
        keys = np.stack([self.rng.choice(self.n_keys, N, replace=False) for _ in range(nb)])
        vals = np.stack([self.rng.choice(self.n_keys, N, replace=False) for _ in range(nb)])
        x[:, self.ctx_start:self.ctx_end:2] = self.KEY0 + keys
        x[:, self.ctx_start + 1:self.ctx_end:2] = self.VAL0 + vals
        # query slots: n_queries positions in [q_lo, T-1), spaced >= 2 apart
        slots = np.arange(self.q_lo, T - 1, 2)
        pairs = np.zeros((nb, self.n_queries, 2), dtype=np.int64)
        for b in range(nb):
            qpos = np.sort(self.rng.choice(slots, self.n_queries, replace=False))
            which = self.rng.integers(0, N, size=self.n_queries)
            x[b, qpos] = self.KEY0 + keys[b, which]
            y[b, qpos] = self.VAL0 + vals[b, which]
            # the state a query must reach is its payload's, one after its key
            pairs[b, :, 0] = qpos
            pairs[b, :, 1] = self.ctx_start + 2 * which + 1
        self.last_pairs = torch.as_tensor(pairs, device=self.device)
        keep = np.zeros((nb, T), dtype=np.float32)          # task tokens: kv pairs and queries
        keep[:, self.ctx_start:self.ctx_end] = 1.0
        keep[y != -100] = 1.0
        self.last_keep = torch.as_tensor(keep, device=self.device)
        return torch.as_tensor(x, device=self.device), torch.as_tensor(y, device=self.device)


def set_oracle(model, keep):
    for b in model.blocks:
        if b.attn.arm == "smat":
            b.attn.key_keep = keep


class ZoologyMQAR:
    """Zoology's ``multiquery_ar`` generator (Arora et al.), reproduced:
    vocab split into keys [1, V/2) and values [V/2, V); ``num_kv_pairs`` pairs
    ``k1 v1 k2 v2 ...`` at the start; every key queried exactly once, at
    position context + 2*gap with gaps drawn without replacement from
    p(gap) ~ gap^(power_a - 1), power_a = 0.01 (queries cluster near the
    context); label = the value at the position after the query key; all
    other positions random tokens from the whole vocabulary."""

    def __init__(self, T, n_pairs, vocab, device, seed=0, power_a=0.01):
        assert T % 2 == 0 and vocab > T and n_pairs * 4 <= T
        self.T, self.n_pairs, self.vocab, self.device = T, n_pairs, vocab, device
        self.rng = np.random.default_rng(seed)
        self.ctx = 2 * n_pairs
        self.kv = vocab // 2
        self.n_queries = n_pairs
        space = (T - self.ctx) // 2
        p = power_a * np.arange(1, space + 1) ** (power_a - 1)
        self.p = p / p.sum()
        self.space = space
        self.ctx_start, self.ctx_end = 0, self.ctx      # for the oracle / pollution probes
        self.q_lo = self.ctx

    def batch(self, nb):
        T, N = self.T, self.n_pairs
        keys = np.stack([self.rng.choice(np.arange(1, self.kv), N, replace=False) for _ in range(nb)])
        vals = np.stack([self.rng.choice(np.arange(self.kv, self.vocab), N, replace=False) for _ in range(nb)])
        ex = np.zeros((nb, T + 1), dtype=np.int64)
        ex[:, 0:self.ctx:2] = keys
        ex[:, 1:self.ctx:2] = vals
        lab = np.full((nb, T + 1), -100, dtype=np.int64)
        for b in range(nb):
            gaps = self.rng.choice(self.space, N, replace=False, p=self.p)
            ex[b, self.ctx + 2 * gaps] = keys[b]
            lab[b, self.ctx + 2 * gaps + 1] = vals[b]
        x, y = np.ascontiguousarray(ex[:, :-1]), np.ascontiguousarray(lab[:, 1:])
        noise = self.rng.integers(0, self.vocab, size=x.shape)
        x = np.ascontiguousarray(np.where(x == 0, noise, x))
        keep = np.zeros_like(x, dtype=np.float32); keep[:, :self.ctx] = 1.0; keep[y != -100] = 1.0
        self.last_keep = torch.as_tensor(keep, device=self.device)
        return torch.as_tensor(x, device=self.device), torch.as_tensor(y, device=self.device)


@torch.no_grad()
def pollution(model, task, x, y):
    """Share of the normaliser at query rows that comes from noise tokens, per layer:
    phi(q_t) . sum_{j<=t, noise} phi(k_j)  /  phi(q_t) . sum_{j<=t} phi(k_j)."""
    out = []
    keep = task.last_keep                                            # (nb, T)
    for b in model.blocks:
        if b.attn.arm != "smat" or b.attn._last is None:
            continue
        Phi, Psi = b.attn._last                                      # (nb*H, T, r) fp32
        H = b.attn.h; nb, T = keep.shape
        kf = keep.unsqueeze(1).expand(nb, H, T).reshape(nb * H, T, 1)
        cum_all = torch.cumsum(Psi, dim=1)
        cum_noise = torch.cumsum(Psi * (1 - kf), dim=1)
        rows = (y != -100)                                           # (nb, T)
        rf = rows.unsqueeze(1).expand(nb, H, T).reshape(nb * H, T)
        num = (Phi * cum_noise).sum(-1)[rf]
        den = (Phi * cum_all).sum(-1)[rf]
        out.append((num / den.clamp_min(1e-12)).mean().item())
    return out


@torch.no_grad()
def hash_hits(model, x, y, n):
    """Prop. "Content-addressed recall", measured: the fraction of query rows
    whose profile cell equals the cell of the key they are looking for.  ~1 means
    the hash is aligned and any remaining failure is downstream (bucket load,
    state rank, the gate); ~1/N0 means the hash never aligned at all."""
    out = []
    for b in model.blocks:
        ca = getattr(b.attn, "ca", None)
        if ca is None or ca.last is None:
            continue
        kc, qc = ca.last                                   # (nb*h, n), (nb*h, T-n)
        nb = x.shape[0]
        h = kc.shape[0] // nb
        hit = tot = 0
        for bb in range(nb):
            for t in torch.nonzero(y[bb] != -100).flatten().tolist():
                i = t - n
                if i < 0:
                    continue                               # query fell in the distant block
                j = torch.nonzero(x[bb, :n] == x[bb, t]).flatten()
                if j.numel() == 0:
                    continue
                j = int(j[0]) + 1          # the payload sits one after its key, and it
                if j >= n:                 # is the payload's state the query must see
                    continue
                for hh in range(h):
                    hit += int(kc[bb * h + hh, j] == qc[bb * h + hh, i])
                    tot += 1
        out.append(hit / max(tot, 1))
    return out

@torch.no_grad()
def evaluate(model, task, n, nb, amp, oracle=False):
    model.eval()
    for b in model.blocks:
        if getattr(b.attn, "ca", None) is not None:
            b.attn.ca.probe = True
    correct = tot = 0; nll = 0.0; poll = []; hits = []
    for i in range(n // nb):
        x, y = task.batch(nb)
        if oracle:
            set_oracle(model, task.last_keep)
        for b in model.blocks:
            b.attn.probe = (i == 0)
        with torch.autocast(device_type=x.device.type, dtype=amp, enabled=amp is not None):
            logits = model(x)
        if i == 0:
            poll = pollution(model, task, x, y)
            nb0 = model.blocks[0].attn.spec.n if getattr(model.blocks[0].attn, "spec", None) is not None else 0
            hits = hash_hits(model, x, y, nb0)
            for b in model.blocks:
                b.attn.probe = False
                if getattr(b.attn, "ca", None) is not None:
                    b.attn.ca.probe = False
        sel = y != -100
        lg = logits[sel].float(); yy = y[sel]
        nll += F.cross_entropy(lg, yy, reduction="sum").item()
        correct += (lg.argmax(-1) == yy).sum().item(); tot += yy.numel()
    model.train()
    return {"acc": correct / tot, "nll": nll / tot,
            "noise_share": " ".join(f"{p:.3f}" for p in poll),
            "hash_hit": " ".join(f"{p:.3f}" for p in hits),
            "cell_occ": " ".join(f"{b.attn.ca.last_occ:.3f}" for b in model.blocks
                                 if getattr(b.attn, "ca", None) is not None
                                 and b.attn.ca.last_occ is not None),
            "g_density": " ".join(f"{b.attn.ca.last_dens:.3f}" for b in model.blocks
                                  if getattr(b.attn, "ca", None) is not None
                                  and b.attn.ca.last_dens is not None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["softmax", "smat", "mamba2", "mamba2smat"], required=True)
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--n-keys", type=int, default=2048)
    ap.add_argument("--ctx-start", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--nb", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--qk-norm", action="store_true")
    ap.add_argument("--short-conv", type=int, default=0, help="causal depthwise conv width before attention (0=off)")
    ap.add_argument("--triton-bwd", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--phi", choices=["elu1", "taylor"], default="elu1")
    ap.add_argument("--decay", choices=["off", "on", "coupled", "mamba2"], default="off")
    ap.add_argument("--dt-min", type=float, default=1e-3)
    ap.add_argument("--dt-max", type=float, default=1e-1)
    ap.add_argument("--A-min", type=float, default=1.0)
    ap.add_argument("--A-max", type=float, default=16.0)
    ap.add_argument("--scan-kernel", choices=["phi", "phi-nonorm", "signed"], default="phi")
    ap.add_argument("--pos-read", choices=["plane", "point"], default="plane",
                    help="positional arm: read a hyperplane of profiles (the published map) or a single profile")
    ap.add_argument("--agree-coef", type=float, default=0.0,
                    help="routing objective: the query's cell distribution should carry mass on its payload's cell")
    ap.add_argument("--anneal-steps", type=int, default=0,
                    help="steps over which the soft top-k read is annealed to the hard mask (0 = hard throughout)")
    ap.add_argument("--hash-src", choices=["hidden", "embed", "id"], default="hidden",
                    help="hash the layer input, or the token embedding (identical for a repeated token)")
    ap.add_argument("--hash-freeze", action="store_true", help="frozen random hash: nothing to learn, cannot collapse")
    ap.add_argument("--hash-conv", type=int, default=0,
                    help="learned depthwise causal conv of this width on the key-side hash input, "
                         "replacing the fixed --hash-shift")
    ap.add_argument("--unfreeze-at", type=int, default=0,
                    help="curriculum: train with the hash frozen, then release it at this step")
    ap.add_argument("--hash-shift", type=int, default=0,
                    help="profile of position j is the hash of position j-shift; MQAR needs 1 (the payload follows its key)")
    ap.add_argument("--balance-coef", type=float, default=0.01,
                    help="weight on the load-balancing penalty of the profile hash")
    ap.add_argument("--assign", choices=["positional", "content"], default="positional",
                    help="how prof/type are computed: fixed round-robin, or hashes of token content")
    ap.add_argument("--read-mode", choices=["point", "plane"], default="point",
                    help="content types: own cell only, or the hyperplane of a hashed direction through it")
    ap.add_argument("--gate-wd", type=float, default=0.1, help="weight decay on gate params (w_dt, A_log, w_a)")
    ap.add_argument("--gate-lr-mult", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--zoology", action="store_true", help="use Zoology's multiquery_ar generator (vocab 8192)")
    ap.add_argument("--n-P", type=int, default=0, help="global channel width (Remark 3.4); -1 = 2*n_pairs")
    ap.add_argument("--early-stop", type=float, default=0.0, help="stop when eval acc >= this (0 = off)")
    ap.add_argument("--mask-bank", action="store_true", help="distinct row-type offset per (layer, head)")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--oracle-keep", action="store_true",
                    help="probe: zero every noise token's key (no state pollution)")
    ap.add_argument("--taylor-dim", type=int, default=16)
    ap.add_argument("--pad-noise", action="store_true", help="fill non-task positions with one PAD token")
    ap.add_argument("--out", default="results/mqar.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    amp = None if (dev.type != "cuda" or args.no_amp) else torch.bfloat16
    if args.triton_bwd:
        from smat.kernels import triton_bwd as smat_triton_bwd
        smat_triton_bwd.patch()
    name = ((args.arm if args.arm not in ("smat", "mamba2smat") else f"{args.arm}_d{args.d}") + f"_T{args.T}_p{args.n_pairs}" + ("_pad" if args.pad_noise else "") + ("_zoo" if args.zoology else "")
            + (f"_taylor{args.taylor_dim}" if args.phi == "taylor" else "") + ("_oracle" if args.oracle_keep else "")
            + (f"_P{2 * args.n_pairs if args.n_P == -1 else args.n_P}" if args.n_P else "") + ("_bank" if args.mask_bank else "") + (f"_{args.scan_kernel}" if args.scan_kernel != "phi" else "") + ("_c" + args.read_mode if args.assign == "content" else "") + ("_ppoint" if args.assign == "positional" and args.pos_read == "point" else "") + (f"_{args.hash_src}" if args.assign == "content" and args.hash_src != "hidden" else "") + ("_frz" if args.hash_freeze else "") + (f"_unfrz{args.unfreeze_at}" if args.unfreeze_at else "") + (f"_sh{args.hash_shift}" if args.hash_shift else "") + (f"_cv{args.hash_conv}" if args.hash_conv else "") + (f"_ag{args.agree_coef:g}" if args.agree_coef else "") + (f"_an{args.anneal_steps}" if args.anneal_steps else "") + f"_dm{args.d_model}_s{args.seed}" + ("" if args.decay == "off" else f"_decay{args.decay if args.decay != 'on' else ''}") + args.tag)
    print(json.dumps({"run": name, **vars(args)}, default=str), flush=True)

    if args.zoology:
        task = ZoologyMQAR(args.T, args.n_pairs, args.vocab, dev, seed=args.seed)
    else:
        task = MQAR(args.T, args.n_pairs, args.n_queries, args.n_keys, dev, seed=args.seed,
                    ctx_start=args.ctx_start, pad_noise=args.pad_noise)
    spec = None
    nP = 2 * args.n_pairs if args.n_P == -1 else args.n_P
    if args.arm in ("smat", "mamba2smat"):
        # the builder needs >= q^(d-1) profiled columns; take the largest feasible global width
        spec_np = None
        while nP >= 0:
            try:
                spec_np = build_mask(args.T, args.d, chunk=args.chunk, n_P=nP); break
            except ValueError:
                nP -= 1
        if nP != (2 * args.n_pairs if args.n_P == -1 else args.n_P):
            print(f"global channel capped at n_P={nP} (requested {2 * args.n_pairs if args.n_P == -1 else args.n_P})", flush=True)
        spec = to_device(spec_np, dev)
        print(f"mask n={spec.n}: pairs occupy [{task.ctx_start},{task.ctx_end}), queries in [{task.q_lo},{args.T})", flush=True)
    model = train_lm.LM(args.T, args.d_model, args.heads, args.layers, args.arm, spec=spec,
                        r=args.r, chunk=args.chunk, device=dev, seed=args.seed,
                        qk_norm=args.qk_norm, triton_bwd=args.triton_bwd, vocab=task.vocab,
                        short_conv=args.short_conv, phi_kind=args.phi, taylor_dim=args.taylor_dim, mask_bank=args.mask_bank,
                        decay=(False if args.decay == "off" else (args.decay if args.decay != "on" else True)),
                        dt_init=(args.dt_min, args.dt_max), A_init=(args.A_min, args.A_max),
                        scan_kernel=args.scan_kernel, assign=args.assign,
                        read_mode=args.read_mode, pos_read=args.pos_read, hash_src=args.hash_src,
                        hash_freeze=args.hash_freeze, hash_shift=args.hash_shift,
                        hash_conv=args.hash_conv).to(dev)
    gate_p = [p for n, p in model.named_parameters() if any(k in n for k in ("w_dt", "A_log", "w_a", "w_h"))]
    gate_ids = {id(p) for p in gate_p}
    rest = [p for p in model.parameters() if id(p) not in gate_ids]
    opt = torch.optim.AdamW([{"params": rest, "weight_decay": args.wd, "lr_mult": 1.0},
                             {"params": gate_p, "weight_decay": args.gate_wd, "lr_mult": args.gate_lr_mult}],
                            lr=args.lr, betas=(0.9, 0.95))
    print(f"gate params: {sum(p.numel() for p in gate_p)}  (wd {args.gate_wd}, lr x{args.gate_lr_mult}, dt init [{args.dt_min}, {args.dt_max}])", flush=True)
    sched = lambda s: (s + 1) / args.warmup if s < args.warmup else \
        0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, args.steps - args.warmup)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["run", "arm", "d", "n_pairs", "n_queries", "n_keys", "step", "train_loss", "acc", "nll",
              "noise_share", "hash_hit", "cell_occ", "g_density", "ms_per_step", "elapsed_s", "d_model", "heads", "r", "n_P", "seed", "state_per_head", "params"]
    new = not os.path.exists(args.out)
    fout = open(args.out, "a", newline=""); w = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader()

    t0 = time.time(); times = []; row = None
    for step in range(1, args.steps + 1):
        if args.unfreeze_at and step == args.unfreeze_at:
            for b in model.blocks:
                ca = getattr(b.attn, "ca", None)
                if ca is not None:
                    for prm in (ca.W, ca.gamma, ca.b) + ((ca.Wd,) if hasattr(ca, "Wd") else ()):
                        prm.requires_grad_(True)
                    ca.freeze = False
            print(f"  [unfroze the hash at step {step}]", flush=True)
        if args.anneal_steps:
            a = min(1.0, step / float(args.anneal_steps))
            for b in model.blocks:
                if getattr(b.attn, "ca", None) is not None:
                    b.attn.ca.anneal = a
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step - 1) * g.get("lr_mult", 1.0)
        x, y = task.batch(args.nb)
        if args.oracle_keep:
            set_oracle(model, task.last_keep)
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, task.vocab), y.view(-1), ignore_index=-100)
        if args.agree_coef and getattr(task, "last_pairs", None) is not None:
            nsp = model.blocks[0].attn.spec.n
            pr = task.last_pairs
            ij = torch.stack([pr[..., 0] - nsp, pr[..., 1]], dim=-1)
            ok = (ij[..., 0] >= 0) & (ij[..., 1] >= 0) & (ij[..., 1] < nsp)
            ag = [b.attn.ca.agreement(ij, ok) for b in model.blocks
                  if getattr(b.attn, "ca", None) is not None]
            ag = [a for a in ag if a is not None]
            if ag:
                loss = loss + args.agree_coef * torch.stack(ag).mean()
        if args.balance_coef:
            # keep the profile hash spread over F_q^{dim}: cost is unaffected by
            # imbalance but Prop. "Content-addressed recall" is not, and a
            # collapsed hash reads the whole distant block into every query.
            aux = [b.attn.ca.aux for b in model.blocks
                   if getattr(b.attn, "ca", None) is not None and b.attn.ca.aux is not None]
            if aux:
                loss = loss + args.balance_coef * torch.stack(aux).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(loss):
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - ts)
        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  {1000*np.mean(times[-100:]):.0f} ms/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, task, args.eval_n, min(args.nb, 16), amp, oracle=args.oracle_keep)
            row = {"run": name, "arm": args.arm, "d": args.d if args.arm in ("smat", "mamba2smat") else "",
                   "n_pairs": args.n_pairs, "n_queries": args.n_queries, "n_keys": args.n_keys,
                   "step": step, "train_loss": loss.item(), **ev,
                   "d_model": args.d_model, "heads": args.heads, "r": args.r, "n_P": nP, "seed": args.seed,
                   "state_per_head": args.r * (args.d_model // args.heads + 1),
                   "params": sum(p.numel() for p in model.parameters()),
                   "ms_per_step": 1000 * float(np.mean(times[-args.eval_every:])),
                   "elapsed_s": time.time() - t0}
            w.writerow(row); fout.flush()
            print(f"== eval step {step}: acc {ev['acc']:.3f}  nll {ev['nll']:.3f}  noise share [{ev['noise_share']}]  hash hit [{ev['hash_hit']}]  cell occ [{ev['cell_occ']}]  G density [{ev['g_density']}]", flush=True)
            if args.early_stop and ev["acc"] >= args.early_stop:
                print(f"early stop at step {step}: acc {ev['acc']:.3f}", flush=True)
                break
    fout.close()
    print("done:", json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()

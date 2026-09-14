"""Streaming decode on a task: accuracy, per-token cost and cache size together.

Every accuracy number in the paper comes from the content-addressed assignment
and every cost exponent from the positional one, so the construction's central
claim -- O(rp) per generated token from a Theta(T^{1-1/d}) cache rather than
O(T) from a Theta(T) one -- has never been measured on a model that is also
answering correctly.  This script closes that: one trained model, run one token
at a time through the decode path, reporting

    exact-match accuracy | microseconds per token | cache words

against softmax with a full KV cache and against linear attention with its
fixed state.

The decode path is reconstructed here rather than taken from
``smat_attn.SmatDecoder``, which predates the decay and the content-addressed
assignment: its step is ``S += psi v^T`` with no gate and its cache is the
positional type table.  Nothing in ``../smat`` is modified.

Correctness gate: ``--mode parity`` runs the full forward and the streaming
decode on the same batch and compares the logits over the recent block.  The
chunked scan and the sequential recurrence are the same sum in a different
order, so they agree to fp32 rounding, not exactly; the gate is a relative
error, and every other mode refuses to run until it passes.

Length: the assignment is content-based, so the mask is rebuilt at evaluation
length and a model trained at one T can be read at another.  ``--eval-T`` is a
sweep, and whether accuracy survives it is itself the experiment.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))          # the smat package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling task modules
from smat.mask import build_mask                                   # noqa: E402
from smat.attention import (featurise, to_device, pool_profiles,        # noqa: E402
                       apply_incidence)                            # noqa: E402
from smat import model as train_lm                                                    # noqa: E402
from mqar import MQAR                                           # noqa: E402

FIELDS = ["arm", "d", "geometry", "train_T", "eval_T", "n", "n_queries", "acc",
          "us_per_token", "cache_words", "kv_cache_words", "cache_ratio",
          "cells", "mask_cells", "mask_B"]


# ---------------------------------------------------------------------------
# the content-addressed long-range branch, split into prefill and step
# ---------------------------------------------------------------------------
# ContentAssign.forward pools the whole landmark block and reads every recent
# row in one shot.  Decoding needs the same arithmetic split in two: the pool is
# a function of the prefix alone and is computed once, and a step contracts one
# query against one (direction, offset) state.  The two functions below mirror
# that forward line for line; the parity gate is what certifies they do.

def ca_prefill(ca, u_pre, Phi_pre, Psi_pre, Vb_pre, ids_pre=None):
    """Pool the landmark block into the decode cache.

    ``u_pre``: (nb, n, d_model) layer input over the prefix.  Returns the cache
    read by ``ca_step``: for a plane read, ``(nb*h, D, q, r, p)`` -- one state
    per (direction, offset), which is the type table of Theorem "decode"; for a
    point read, ``(nb*h, N0, r, p)``.
    """
    if ca.src == "id":
        assert ids_pre is not None, "the id hash needs token ids"
        ik = ids_pre if not ca.shift else torch.cat(
            [ids_pre.new_zeros(ids_pre.shape[0], ca.shift), ids_pre[:, :-ca.shift]], dim=1)
        Wk = ca._cells_id(ik)
    else:
        hk = u_pre
        if ca.conv_width > 1:
            w = ca.conv_width - 1
            hk = ca.kconv(F.pad(u_pre, (0, 0, w, 0)).transpose(1, 2)).transpose(1, 2)
        elif ca.shift:
            hk = torch.cat([u_pre.new_zeros(u_pre.shape[0], ca.shift, u_pre.shape[2]),
                            u_pre[:, :-ca.shift]], dim=1)
        Wk = ca._cells(hk, plant_at=True)                          # (nb*h, n, N0)
    # F_x = sum_{j in cell x} psi_j vb_j^T
    Fc = torch.stack([(Psi_pre * Wk[:, :, x:x + 1]).transpose(-1, -2) @ Vb_pre
                      for x in range(ca.N0)], dim=1)               # (nb*h, N0, r, p)
    if ca.mode == "point":
        return Fc
    return torch.einsum("eox,bxrp->beorp", ca.M, Fc)               # (nb*h, D, q, r, p)


def ca_step(ca, cache, u_t, Phi_t, ids_t=None, dir_ovr=None):
    """One recent row's long-range read.  O(rp): the cache is indexed, not scanned.

    ``u_t``: (nb, 1, d_model).  ``Phi_t``: (nb*h, 1, r).  Returns (nb*h, 1, p).
    """
    if ca.src == "id":
        Wq = ca._cells_id(ids_t)                                   # (nb*h, 1, N0)
    else:
        Wq = ca._cells(u_t)
    if ca.mode == "point":
        Z = torch.einsum("bir,bxrp->bixp", Phi_t, cache)           # (nbh, 1, N0, p)
        return torch.einsum("bix,bixp->bip", Wq, Z)                # (nbh, 1, p)
    nbh = Phi_t.shape[0]
    # direction: the oracle's, or the argmax of the learned logits (the forward
    # path is a hard one-hot; the straight-through term vanishes under no_grad)
    dl = torch.einsum("blm,hem->bhle", u_t, ca.Wd).reshape(nbh, 1, ca.D)
    e = (dir_ovr.reshape(nbh, 1) if dir_ovr is not None else dl.argmax(-1))   # (nbh, 1)
    # offset: the cell the query lands in fixes which plane of that direction
    po = torch.einsum("bix,eox->bieo", Wq, ca.M)                   # (nbh, 1, D, q)
    po_e = po.gather(2, e.view(nbh, 1, 1, 1).expand(nbh, 1, 1, ca.q)).squeeze(2)
    o = po_e.argmax(-1)                                            # (nbh, 1)
    r, p = cache.shape[-2], cache.shape[-1]
    idx = (e * ca.q + o).view(nbh, 1, 1, 1).expand(nbh, 1, r, p)
    U_eo = cache.reshape(nbh, ca.D * ca.q, r, p).gather(1, idx)    # (nbh, 1, r, p)
    return torch.einsum("bir,birp->bip", Phi_t, U_eo)              # (nbh, 1, p)


def pos_prefill(at, Phi_pre, Psi_pre, Vb_pre):
    """Type table for the published positional maps: ``(nb*h, B, r, p)``.

    This is the cache ``SmatDecoder`` builds, recomputed here so the decay and
    the rest of the block stay on one code path.  Row ``i`` of the recent block
    reads ``U[rho(i)]`` with ``rho(i) = i mod B``, as ``query_by_type`` does.
    """
    spec = at.spec
    if spec is None or spec.n >= Phi_pre.shape[1] + 1:
        return None
    assert at.type_offsets is None, "the mask bank is not supported by this decoder"
    assert not getattr(at, "pos_point", False), "pos_read=point is not supported here"
    if spec.n_X > 0:
        Fp = pool_profiles(Psi_pre, Vb_pre, spec, acc_dtype=torch.float32, backend=at.backend)
        U = apply_incidence(Fp, spec, backend=at.backend)
    else:
        U = torch.zeros(Phi_pre.shape[0], spec.B, Phi_pre.shape[-1], Vb_pre.shape[-1],
                        device=Phi_pre.device, dtype=torch.float32)
    if spec.n_P:
        nP = spec.n_P
        U = U + (Psi_pre[:, :nP].transpose(-1, -2) @ Vb_pre[:, :nP]).unsqueeze(1)
    return U


# ---------------------------------------------------------------------------
# streaming decoder for a whole trained LM
# ---------------------------------------------------------------------------
class StreamDecoder:
    """Runs ``train_lm.LM`` one token at a time over the recent block.

    Prefill is a single forward over ``[0, n)`` that stashes, per layer, the
    pooled cache and the short convolution's ring buffer.  A step costs O(rp)
    per layer plus the pointwise parts of the block, independent of T.
    """

    def __init__(self, model, x_pre):
        self.model = model
        self.L = len(model.blocks)
        self.caches, self.S, self.ring = [], [], []
        self.arm = model.blocks[0].attn.arm
        self.i = 0                      # index of the next row inside the recent block
        self.kv = []                    # softmax arm: (K, V) per layer
        model.eval()
        with torch.no_grad():
            self._prefill(x_pre)

    # -- prefill ----------------------------------------------------------
    def _prefill(self, x_pre):
        m = self.model
        nb, n = x_pre.shape
        emb = m.tok(x_pre)
        x = emb + m.pos(torch.arange(n, device=x_pre.device))
        self.n = n
        for blk in m.blocks:
            at = blk.attn
            # the block's own input is what the hash and the conv see
            if blk.conv is not None:
                lnx = blk.ln0(x)
                w = blk.conv.kernel_size[0]
                self.ring.append(lnx[:, -(w - 1):].clone() if w > 1
                                 else lnx.new_zeros(nb, 0, lnx.shape[-1]))
                x = x + blk.conv(lnx.transpose(1, 2))[..., :n].transpose(1, 2)
            else:
                self.ring.append(None)
            u = blk.ln1(x)
            if self.arm == "softmax":
                q, k, v = at.qkv(u).view(nb, n, 3, at.h, at.dh).unbind(2)
                q, k = at.qn(q), at.kn(k)
                self.kv.append([k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()])
                o = F.scaled_dot_product_attention(q.transpose(1, 2), self.kv[-1][0],
                                                   self.kv[-1][1], is_causal=True).transpose(1, 2)
                self.caches.append(None); self.S.append(None)
                x = x + at.out(o.reshape(nb, n, -1))
            else:
                q, k, v = at.qkv(u).view(nb, n, 3, at.h, at.dh).unbind(2)
                q, k = at.qn(q), at.kn(k)
                fold = lambda t: t.transpose(1, 2).reshape(nb * at.h, n, at.dh).float()
                Phi, Psi, Vb = featurise(fold(q), fold(k), fold(v), at.phi)
                ca = getattr(at, "ca", None)
                if ca is not None:
                    self.caches.append(ca_prefill(ca, u.float(), Phi, Psi, Vb, ids_pre=x_pre))
                else:
                    # positional assignment (the d=1 reference and the published
                    # maps): one state per row type, read by rho(i) = i mod B
                    self.caches.append(pos_prefill(at, Phi, Psi, Vb))
                # the scan resets at n, so the recent state starts empty and the
                # prefix's own outputs are never needed again
                r, p = Phi.shape[-1], Vb.shape[-1]
                self.S.append(torch.zeros(nb * at.h, r, p, device=Phi.device, dtype=torch.float32))
                H = at._scan(Phi, Psi, Vb, u.float(), nb, n)
                o = H[..., :-1] / H[..., -1:].clamp_min(1e-6)
                o = o.reshape(nb, at.h, n, at.dh).transpose(1, 2).reshape(nb, n, -1)
                x = x + at.out(o)
            x = x + blk.mlp(blk.ln2(x))
        self.x_last = x

    # -- one token --------------------------------------------------------
    def step(self, tok, dir_ovr=None):
        """``tok``: (nb,) token ids.  Returns logits (nb, vocab)."""
        m = self.model
        nb = tok.shape[0]
        pos = torch.full((nb,), self.n, device=tok.device, dtype=torch.long)
        x = (m.tok(tok) + m.pos(pos)).unsqueeze(1)                 # (nb, 1, d_model)
        for li, blk in enumerate(m.blocks):
            at = blk.attn
            if blk.conv is not None:
                lnx = blk.ln0(x)
                w = blk.conv.kernel_size[0]
                buf = torch.cat([self.ring[li], lnx], dim=1) if w > 1 else lnx
                # the conv pads both sides; the full forward keeps only the first
                # L outputs, so the row for the newest token is at L-1, not -1
                y = blk.conv(buf.transpose(1, 2)).transpose(1, 2)[:, :buf.shape[1]][:, -1:]
                if w > 1:
                    self.ring[li] = buf[:, -(w - 1):]
                x = x + y
            u = blk.ln1(x)
            q, k, v = at.qkv(u).view(nb, 1, 3, at.h, at.dh).unbind(2)
            q, k = at.qn(q), at.kn(k)
            if self.arm == "softmax":
                K, V = self.kv[li]
                K = torch.cat([K, k.transpose(1, 2)], dim=2)
                V = torch.cat([V, v.transpose(1, 2)], dim=2)
                self.kv[li] = [K, V]
                o = F.scaled_dot_product_attention(q.transpose(1, 2), K, V).transpose(1, 2)
                x = x + at.out(o.reshape(nb, 1, -1))
            else:
                fold = lambda t: t.transpose(1, 2).reshape(nb * at.h, 1, at.dh).float()
                Phi, Psi, Vb = featurise(fold(q), fold(k), fold(v), at.phi)
                uf = u.float()
                if at.decay == "mamba2":
                    dt = F.softplus(at.w_dt(uf))                   # (nb, 1, H)
                    logA = -(dt * torch.exp(at.A_log).float())
                    a = torch.exp(logA).transpose(1, 2).reshape(nb * at.h, 1, 1)
                    dtf = dt.transpose(1, 2).reshape(nb * at.h, 1, 1)
                    Psi = Psi * dtf
                elif at.decay:
                    logA = F.logsigmoid(at.w_a(uf).float()) / 16.0
                    a = torch.exp(logA).transpose(1, 2).reshape(nb * at.h, 1, 1)
                else:
                    a = torch.ones(nb * at.h, 1, 1, device=Phi.device)
                self.S[li] = a * self.S[li] + Psi.transpose(-1, -2) @ Vb
                H = Phi @ self.S[li]                               # (nb*h, 1, p)
                ca = getattr(at, "ca", None)
                if ca is not None:
                    H = H + ca_step(ca, self.caches[li], uf, Phi, ids_t=tok.unsqueeze(1),
                                    dir_ovr=dir_ovr)
                elif self.caches[li] is not None:
                    U = self.caches[li]
                    H = H + (Phi @ U[:, self.i % U.shape[1]])
                o = H[..., :-1] / H[..., -1:].clamp_min(1e-6)
                o = o.reshape(nb, at.h, 1, at.dh).transpose(1, 2).reshape(nb, 1, -1)
                x = x + at.out(o)
            x = x + blk.mlp(blk.ln2(x))
        self.n += 1
        self.i += 1
        return m.head(m.ln_f(x))[:, 0]

    # -- accounting -------------------------------------------------------
    def cache_words(self):
        if self.arm == "softmax":
            return int(sum(K.numel() + V.numel() for K, V in self.kv))
        w = sum(int(S.numel()) for S in self.S if S is not None)
        w += sum(int(c.numel()) for c in self.caches if c is not None)
        return w


# ---------------------------------------------------------------------------
# model construction, mirroring mqar_lm so a checkpoint is interchangeable
# ---------------------------------------------------------------------------
def build(args, T, vocab, dev):
    spec = None
    if args.arm == "smat":
        nP = 2 * args.n_pairs
        while nP >= 0:
            try:
                spec = build_mask(T, args.d, chunk=args.chunk, n_P=nP); break
            except ValueError:
                nP -= 1
        spec = to_device(spec, dev)
    model = train_lm.LM(T, args.d_model, args.heads, args.layers, args.arm, spec=spec,
                        r=args.r, chunk=args.chunk, device=dev, seed=args.seed,
                        qk_norm=True, vocab=vocab, short_conv=args.short_conv,
                        decay="mamba2", dt_init=(args.dt_min, args.dt_max),
                        A_init=(1.0, 16.0), assign=("content" if args.d >= 2 else "positional"),
                        read_mode=args.read_mode, hash_src=args.hash_src,
                        hash_freeze=True, hash_shift=args.hash_shift).to(dev)
    return model, spec


def task_for(args, T, dev):
    return MQAR(T, args.n_pairs, args.n_queries, args.n_keys, dev, seed=args.seed,
                ctx_start=0)


# ---------------------------------------------------------------------------
# the correctness gate
# ---------------------------------------------------------------------------
def parity(args, dev):
    """Full forward vs streaming decode on the same batch, over the recent block."""
    torch.manual_seed(args.seed)
    task = task_for(args, args.T, dev)
    model, spec = build(args, args.T, task.vocab, dev)
    model.eval()
    x, _ = task.batch(args.nb)
    n = spec.n if spec is not None else args.T // 2
    with torch.no_grad():
        ref = model(x)[:, n:]                                      # (nb, T-n, vocab)
        dec = StreamDecoder(model, x[:, :n])
        got = torch.stack([dec.step(x[:, t]) for t in range(n, args.T)], dim=1)
    num = (ref - got).abs().max().item()
    den = ref.abs().max().item()
    rel = num / max(den, 1e-9)
    agree = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
    print(json.dumps({"mode": "parity", "arm": args.arm, "d": args.d, "T": args.T, "n": n,
                      "max_abs_diff": num, "logit_scale": den, "rel_err": rel,
                      "argmax_agreement": agree}), flush=True)
    ok = rel < args.parity_tol and agree > 0.999
    print(("PARITY OK" if ok else "PARITY FAILED")
          + f": rel_err {rel:.3e} (tol {args.parity_tol}), argmax agreement {agree:.4f}",
          flush=True)
    return ok


# ---------------------------------------------------------------------------
# training (the same recipe as mqar_lm, kept short: this script is not the place
# to re-tune it)
# ---------------------------------------------------------------------------
def train(args, dev):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    task = task_for(args, args.T, dev)
    model, spec = build(args, args.T, task.vocab, dev)
    gate = [p for nm, p in model.named_parameters() if any(k in nm for k in ("w_dt", "A_log"))]
    gid = {id(p) for p in gate}
    rest = [p for p in model.parameters() if id(p) not in gid]
    opt = torch.optim.AdamW([{"params": rest, "weight_decay": args.wd},
                             {"params": gate, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    warm = 200
    sched = lambda s: (s + 1) / warm if s < warm else \
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm)))
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step - 1)
        x, y = task.batch(args.nb)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            acc = accuracy_full(model, task, args.nb, 4)
            print(f"== step {step}: loss {loss.item():.4f}  acc {acc:.4f}  "
                  f"{1000 * (time.time() - t0) / step:.0f} ms/step", flush=True)
            if acc >= args.early_stop:
                print(f"early stop at {acc:.4f}", flush=True)
                break
    torch.save({"sd": model.state_dict(), "vocab": task.vocab, "args": vars(args)}, args.ckpt)
    print(f"checkpoint -> {args.ckpt}", flush=True)
    return model


@torch.no_grad()
def accuracy_full(model, task, nb, iters):
    model.eval(); ok = tot = 0
    for _ in range(iters):
        x, y = task.batch(nb)
        pred = model(x).argmax(-1)
        m = y != -100
        ok += (pred[m] == y[m]).sum().item(); tot += int(m.sum().item())
    model.train()
    return ok / max(tot, 1)


# ---------------------------------------------------------------------------
# the measurement: accuracy, per-token cost and cache size at one length
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure(model, args, T, dev, w, geometry="trained"):
    """Decode the recent block one token at a time at evaluation length ``T``."""
    task = task_for(args, T, dev)
    # the mask is rebuilt at this length; a content-addressed assignment does not
    # depend on absolute position, so the trained weights are reusable
    spec = None
    if args.arm == "smat":
        nP = 2 * args.n_pairs
        while nP >= 0:
            try:
                spec = build_mask(T, args.d, chunk=args.chunk, n_P=nP); break
            except ValueError:
                nP -= 1
        spec = to_device(spec, dev)
        for blk in model.blocks:
            blk.attn.spec = spec
            if getattr(blk.attn, "ca", None) is not None:
                blk.attn.ca.spec = spec
    n = spec.n if spec is not None else T // 2
    # positional embeddings are trained at args.T; beyond it there is nothing to
    # read, so the table is extended by repeating its last row rather than
    # failing.  Any accuracy past args.T is therefore a statement about the mask,
    # not about positions.
    _extend_pos(model, T, dev)

    model.eval()
    x, y = task.batch(args.nb)
    dec = StreamDecoder(model, x[:, :n])
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    preds = []
    for t in range(n, T):
        preds.append(dec.step(x[:, t]))
    if dev == "cuda":
        torch.cuda.synchronize()
    us = 1e6 * (time.time() - t0) / max(1, (T - n) * args.nb)
    pred = torch.stack(preds, dim=1).argmax(-1)
    m = y[:, n:] != -100
    acc = ((pred[m] == y[:, n:][m]).sum().item() / max(1, int(m.sum().item())))
    ca0 = getattr(model.blocks[0].attn, "ca", None)
    row = {"arm": args.arm, "d": args.d, "geometry": geometry, "train_T": args.T,
           "eval_T": T, "n": n, "n_queries": int(m.sum().item()), "acc": acc,
           "us_per_token": us, "cache_words": dec.cache_words(),
           "kv_cache_words": int(args.nb * 2 * T * args.d_model * args.layers),
           "cells": (None if ca0 is None else ca0.N0),
           "mask_cells": (None if spec is None else int(spec.N0)),
           "mask_B": (None if spec is None else int(spec.B))}
    row["cache_ratio"] = row["cache_words"] / row["kv_cache_words"]
    print(f"  T={T:>7} [{geometry}]  acc {acc:.4f}  {us:8.1f} us/token  "
          f"cache {row['cache_words']:>12,} ({row['cache_ratio']:.4f} x KV)", flush=True)
    w.writerow(row)
    return row


@torch.no_grad()
def cost_sweep(args, dev, w):
    """Per-token cost and cache size at the geometry each length actually has.

    Accuracy needs a model trained at that length; cost and cache do not, since
    neither depends on the weights.  Separating them is what lets this sweep run
    to lengths no one can afford to train at, which is where the claim lives.
    The accuracy column is left empty here on purpose -- reading it as a result
    would be reading an untrained model.
    """
    for T in args.eval_T:
        try:
            torch.manual_seed(args.seed)
            task = task_for(args, T, dev)
            model, spec = build(args, T, task.vocab, dev)
            model.eval()
            n = spec.n if spec is not None else T // 2
            x, _ = task.batch(args.nb)
            dec = StreamDecoder(model, x[:, :n])
            steps = min(args.cost_steps, T - n)
            for t in range(n, n + min(8, steps)):        # warm up the kernels
                dec.step(x[:, t])
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for t in range(n, n + steps):
                dec.step(x[:, t % T])
            if dev == "cuda":
                torch.cuda.synchronize()
            us = 1e6 * (time.time() - t0) / max(1, steps * args.nb)
            ca0 = getattr(model.blocks[0].attn, "ca", None)
            row = {"arm": args.arm, "d": args.d, "geometry": "native", "train_T": "",
                   "eval_T": T, "n": n, "n_queries": "", "acc": "", "us_per_token": us,
                   "cache_words": dec.cache_words(),
                   "kv_cache_words": int(args.nb * 2 * T * args.d_model * args.layers),
                   "cells": (None if ca0 is None else ca0.N0),
                   "mask_cells": (None if spec is None else int(spec.N0)),
                   "mask_B": (None if spec is None else int(spec.B))}
            row["cache_ratio"] = row["cache_words"] / row["kv_cache_words"]
            print(f"  T={T:>7} [native]  {us:8.1f} us/token  "
                  f"cache {row['cache_words']:>12,} ({row['cache_ratio']:.4f} x KV)", flush=True)
            w.writerow(row)
            del model, dec
            if dev == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  T={T}: FAILED {type(e).__name__}: {e}", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()


def _extend_pos(model, T, dev):
    old = model.pos.weight.shape[0]
    if T <= old:
        return
    with torch.no_grad():
        new = nn.Embedding(T, model.pos.weight.shape[1]).to(dev)
        new.weight[:old] = model.pos.weight
        new.weight[old:] = model.pos.weight[-1:]
        model.pos = new


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["parity", "cost", "run"], default="run")
    for a, t, d in (("--T", int, 1024), ("--chunk", int, 256), ("--d", int, 2),
                    ("--n-pairs", int, 16), ("--n-queries", int, 16), ("--n-keys", int, 128),
                    ("--layers", int, 4), ("--d-model", int, 256), ("--heads", int, 4),
                    ("--r", int, 64), ("--steps", int, 4000), ("--nb", int, 32),
                    ("--eval-every", int, 500), ("--seed", int, 0), ("--short-conv", int, 3),
                    ("--hash-shift", int, 1)):
        ap.add_argument(a, type=t, default=d)
    ap.add_argument("--arm", choices=["smat", "softmax"], default="smat")
    ap.add_argument("--read-mode", choices=["point", "plane"], default="plane")
    ap.add_argument("--hash-src", choices=["hidden", "embed", "id"], default="id")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--dt-min", type=float, default=0.1)
    ap.add_argument("--dt-max", type=float, default=1.0)
    ap.add_argument("--early-stop", type=float, default=0.995)
    ap.add_argument("--parity-tol", type=float, default=2e-3)
    ap.add_argument("--eval-T", type=int, nargs="+", default=[1024])
    ap.add_argument("--cost-steps", type=int, default=256)
    ap.add_argument("--ckpt", default="ckpt/stream.pt")
    ap.add_argument("--out", default="results/stream.csv")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(json.dumps({**vars(args), "device": dev}), flush=True)

    if args.mode == "parity":
        sys.exit(0 if parity(args, dev) else 1)

    if not parity(args, dev):
        print("refusing to measure: the streaming decode does not reproduce the "
              "full forward, so any cost number would be for a different model",
              flush=True)
        sys.exit(1)

    if args.mode == "cost":
        new = not os.path.exists(args.out)
        f = open(args.out, "a", newline="")
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        cost_sweep(args, dev, w)
        f.close()
        return

    model = train(args, dev)
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    for T in args.eval_T:
        try:
            measure(model, args, T, dev, w, geometry=("trained" if T == args.T else "frozen-cells"))
            f.flush()
        except Exception as e:                                     # OOM at the long end
            print(f"  T={T}: FAILED {type(e).__name__}: {e}", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()
    f.close()


if __name__ == "__main__":
    main()

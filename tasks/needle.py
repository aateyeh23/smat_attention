"""Positional needle-in-a-haystack through the LM stack of train_lm.py.

Haystack: T bytes of enwik8.  Needles: k payload symbols (from an alphabet of P
extra tokens) planted at the k marked landmark columns of the SMAT mask under
test, chosen by the repo's ``choose_marked_set``.  Request: a token ``REQ_m`` at
the recent row ``t_m`` whose mask pattern isolates marker m (``pattern_table``),
so the row sees needle m and no other needle.  Target: the payload symbol at
marker m, predicted from the hidden state at row ``t_m``.

Every arm gets identical sequences.  SMAT at the reference d has the routing
in the mask; d=1 sees all k needles from that row and has to separate them with
a linear kernel; softmax has to learn to attend to position p_m from REQ_m.
Filler values are real text, so the model must also learn to suppress them.
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
from routing_ceiling import mask_columns                        # noqa: E402
from smat import model as train_lm                                               # noqa: E402

BYTES = 256
MARK_REQ0 = BYTES            # REQ_m = BYTES + m
PAY0 = BYTES + 8             # payload symbols PAY_s = PAY0 + s
P = 16
PAD = PAY0 + P               # constant haystack token (--pad-haystack)
VOCAB = PAD + 1


class NeedleTask:
    def __init__(self, T, d_ref, k, data_path, device, seed=0, chunk=128, pad=False):
        self.T, self.k, self.device, self.pad = T, k, device, pad
        self.rng = np.random.default_rng(seed)
        spec = build_mask(T, d_ref, chunk=chunk)
        self.cols, self.rows = self._choose(spec, k)
        raw = np.fromfile(data_path, dtype=np.uint8)
        self.train = raw[:90_000_000]
        self.val = raw[90_000_000:95_000_000]

    def _choose(self, spec, k, tries=500):
        """k landmark columns with distinct profiles, and for each a recent row
        whose allowed landmark set contains that column and none of the others
        (pattern == singleton {m} over the marked set)."""
        n, n_P = spec.n, spec.n_P
        prof = spec.prof_of_column()
        for _ in range(tries):
            cols = np.sort(self.rng.choice(np.arange(n_P, n), k, replace=False))
            if len({int(prof[c - n_P]) for c in cols}) < k:
                continue
            sup = mask_columns(spec, cols)[n:]                       # recent rows only
            packed = sup.astype(np.int64) @ (1 << np.arange(k, dtype=np.int64))
            rows = []
            for m in range(k):
                hit = np.flatnonzero(packed == (1 << m))
                if hit.size == 0:
                    break
                rows.append(int(n + self.rng.choice(hit)))
            if len(rows) == k:
                return [int(c) for c in cols], rows
        raise ValueError(f"no marked set with isolable singletons found for d={spec.d}, k={k}")

    def batch(self, nb, split="train"):
        if self.pad:
            x = np.full((nb, self.T), PAD, dtype=np.int64)
        else:
            src = self.train if split == "train" else self.val
            starts = self.rng.integers(0, len(src) - self.T - 1, size=nb)
            x = np.stack([src[s:s + self.T] for s in starts]).astype(np.int64)
        pay = self.rng.integers(0, P, size=(nb, self.k))
        m = self.rng.integers(0, self.k, size=nb)
        for l, c in enumerate(self.cols):
            x[:, c] = PAY0 + pay[:, l]
        rows = np.array([self.rows[i] for i in m])
        x[np.arange(nb), rows] = MARK_REQ0 + m
        y = PAY0 + pay[np.arange(nb), m]
        dev = self.device
        return (torch.as_tensor(x, device=dev), torch.as_tensor(rows, device=dev),
                torch.as_tensor(y, device=dev), torch.as_tensor(m, device=dev))


@torch.no_grad()
def evaluate(model, task, n, nb, amp):
    model.eval()
    correct = torch.zeros(task.k); count = torch.zeros(task.k); nll = 0.0
    for _ in range(n // nb):
        x, rows, y, m = task.batch(nb, "val")
        with torch.autocast(device_type=x.device.type, dtype=amp, enabled=amp is not None):
            logits = model(x)
        lg = logits[torch.arange(len(x)), rows].float()
        nll += F.cross_entropy(lg, y, reduction="sum").item()
        hit = (lg.argmax(-1) == y).float().cpu()
        for i in range(task.k):
            sel = (m.cpu() == i)
            correct[i] += hit[sel].sum(); count[i] += sel.sum()
    model.train()
    acc = (correct / count.clamp(min=1)).tolist()
    return {"acc": float(correct.sum() / count.sum()), "nll": nll / count.sum().item(),
            **{f"acc_m{i}": a for i, a in enumerate(acc)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["softmax", "smat"], required=True)
    ap.add_argument("--d", type=int, default=1, help="SMAT mask dimension of this arm")
    ap.add_argument("--d-ref", type=int, default=2, help="mask whose marked set / rows define the task")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--T", type=int, default=4096)
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
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--qk-norm", action="store_true")
    ap.add_argument("--pad-haystack", action="store_true", help="constant PAD haystack instead of enwik8 text")
    ap.add_argument("--triton-bwd", action="store_true")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "enwik8"))
    ap.add_argument("--out", default="results/needle.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    amp = None if dev.type != "cuda" else torch.bfloat16
    if args.triton_bwd:
        import smat_triton_bwd
        smat_triton_bwd.patch()
    name = ((args.arm if args.arm == "softmax" else f"smat_d{args.d}") + f"_ref{args.d_ref}_k{args.k}"
            + ("_pad" if args.pad_haystack else ""))
    print(json.dumps({"run": name, **vars(args)}, default=str), flush=True)

    task = NeedleTask(args.T, args.d_ref, args.k, args.data, dev, seed=args.seed, chunk=args.chunk,
                      pad=args.pad_haystack)
    print(f"needles at {task.cols}, query rows {task.rows}", flush=True)

    spec = None
    if args.arm == "smat":
        spec = to_device(build_mask(args.T, args.d, chunk=args.chunk), dev)
    model = train_lm.LM(args.T, args.d_model, args.heads, args.layers, args.arm, spec=spec,
                        r=args.r, chunk=args.chunk, device=dev, seed=args.seed,
                        qk_norm=args.qk_norm, triton_bwd=args.triton_bwd, vocab=VOCAB).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = lambda s: (s + 1) / args.warmup if s < args.warmup else \
        0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, args.steps - args.warmup)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["run", "arm", "d", "d_ref", "k", "step", "train_loss", "acc", "nll",
              *[f"acc_m{i}" for i in range(args.k)], "ms_per_step", "elapsed_s", "cols", "rows"]
    new = not os.path.exists(args.out)
    fout = open(args.out, "a", newline=""); w = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader()

    t0 = time.time(); times = []; row = None
    for step in range(1, args.steps + 1):
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step - 1)
        x, rows, y, m = task.batch(args.nb)
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            logits = model(x)
        loss = F.cross_entropy(logits[torch.arange(len(x)), rows].float(), y)
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
            ev = evaluate(model, task, args.eval_n, min(args.nb, 16), amp)
            row = {"run": name, "arm": args.arm, "d": args.d if args.arm == "smat" else "",
                   "d_ref": args.d_ref, "k": args.k, "step": step, "train_loss": loss.item(), **ev,
                   "ms_per_step": 1000 * float(np.mean(times[-args.eval_every:])),
                   "elapsed_s": time.time() - t0, "cols": " ".join(map(str, task.cols)),
                   "rows": " ".join(map(str, task.rows))}
            w.writerow(row); fout.flush()
            print(f"== eval step {step}: acc {ev['acc']:.3f}  nll {ev['nll']:.3f}  per-needle "
                  + " ".join(f"{ev[f'acc_m{i}']:.2f}" for i in range(args.k)), flush=True)
    fout.close()
    print("done:", json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()

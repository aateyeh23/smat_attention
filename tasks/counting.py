"""Multi-group counting: an AGGREGATION query rather than a retrieval one.

M counted items sit at random positions in the distant block.  An item is a PAIR:
its symbol at position p, a generic MARK carrying the unit payload at p+1.  A
query names k symbols and must emit how many items carry any of them.

Two properties are deliberate, and the first version of this task had neither.

  * The symbol is OFFSET from the thing being counted.  With the symbol sitting
    at the counted position, a key feature encodes group membership directly, the
    query can hold the whole requested set as a sum of symbol features, and
    phi_q . psi_j is already the indicator of membership -- so linear attention
    counts a union of k groups in one read with no mask at all, which is what it
    did (0.990 at k=1, 0.998 at k=2, against our masked d=2 at 0.320).  The mask
    was a handicap there, restricting a query to one cell when the kernel alone
    could see all of them.  Offsetting forces the same composition recall needs.

  * The alphabet is much larger than the feature rank.  Symbols are drawn from a
    large vocabulary and each sequence uses a small random pool of them, so no
    fixed set of near-orthogonal features per symbol fits in rank r and the
    sum-of-features shortcut degrades.

Why this is the right second test.  Multi-item recall shows d binding on a task
that RETRIEVES a set.  The general condition should be broader: d is the number
of context items a layer can select independently, so it should bind wherever
one position must compute a function of a union of k content-defined groups,
whatever that function is.  Counting is the simplest such function and shares no
machinery with retrieval -- no payload identity, no support readout, and the
answer is a single integer rather than a set.

Geometry, unchanged: the query's own symbol fixes its cell, and the k symbols
share a hyperplane exactly when k <= d-1.  So the same staircase is predicted:

    d=2  k=1     d=3  k<=2     d=4  k<=3

Layout, with the offset made concrete.  A counted item is (symbol at p, MARK at
p+1), and the unit payload sits on the MARK.  An answer block is the k requested
symbols at t-k..t-1 in reverse, so Q[0] lands at t-1, then QMARK at t.  With
hash_shift=1 every row is filed under the token before it, so both the MARK and
the answer row inherit a SYMBOL's cell, and nothing is filed under a token that
also names it.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))          # the smat package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling task modules
from smat.mask import build_mask                                   # noqa: E402
from smat.attention import to_device                                    # noqa: E402
from smat import model as train_lm                                                    # noqa: E402
from mkar import set_oracle_dirs                                # noqa: E402


class Counting:
    def __init__(self, T, n, n_marks, n_syms, k, n_ans, vocab, device, seed=0, pool=8):
        assert 2 * n_marks < n and n_ans * (k + 2) < T - n and pool >= k
        self.T, self.n, self.M, self.S, self.k, self.n_ans = T, n, n_marks, n_syms, k, n_ans
        self.pool, self.vocab, self.device = pool, vocab, device
        self.rng = np.random.default_rng(seed)
        self.SYM0 = 1
        self.MARK = 1 + n_syms                      # one generic token at every counted position
        self.QMARK = 2 + n_syms                     # and one at every answer row
        self.NOISE0 = self.QMARK + 1
        self.cmax = n_marks

    def batch(self, nb):
        T, n, k = self.T, self.n, self.k
        x = self.rng.integers(self.NOISE0, self.vocab, size=(nb, T))
        pay = np.zeros((nb, T, 1), dtype=np.float32)
        tgt = np.zeros((nb, self.n_ans), dtype=np.int64)
        rows = np.zeros((nb, self.n_ans), dtype=np.int64)
        req = np.zeros((nb, self.n_ans, k), dtype=np.int64)
        astep = (T - n) // self.n_ans
        for b in range(nb):
            pool = self.rng.choice(self.S, self.pool, replace=False)   # per-sequence symbol pool
            pos = self.rng.choice(np.arange(0, n - 1, 2), self.M, replace=False)
            syms = pool[self.rng.integers(0, self.pool, size=self.M)]
            x[b, pos] = self.SYM0 + syms               # the symbol names the item ...
            x[b, pos + 1] = self.MARK                  # ... and the MARK is what gets counted
            pay[b, pos + 1, 0] = 1.0
            for m in range(self.n_ans):
                t = n + astep * m + astep - 1
                Q = pool[self.rng.choice(self.pool, k, replace=False)]
                x[b, t - k:t] = self.SYM0 + Q[::-1]    # Q[0] at t-1, so it fixes the row's cell
                x[b, t] = self.QMARK
                rows[b, m] = t
                req[b, m] = self.SYM0 + Q
                tgt[b, m] = int(np.isin(syms, Q).sum())
        t_ = lambda a, d=None: torch.as_tensor(a, device=self.device, dtype=d)
        return (t_(x), t_(pay, torch.float32), t_(tgt), t_(rows), t_(req))


class CountModel(nn.Module):
    def __init__(self, T, d_model, heads, layers, n_classes, vocab, arm="smat", **kw):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(T, d_model)
        self.pay_in = nn.Linear(1, d_model, bias=False)
        self.blocks = nn.ModuleList([train_lm.Block(d_model, heads, arm, layer_idx=i,
                                                    vocab=vocab, **kw) for i in range(layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, idx, pay):
        emb = self.tok(idx)
        x = emb + self.pos(torch.arange(idx.shape[1], device=idx.device)) + self.pay_in(pay)
        for b in self.blocks:
            x = b(x, emb=emb, ids=idx)
        return self.head(self.ln_f(x))


def evaluate(model, task, n, nb, iters, oracle):
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(iters):
            x, pay, tgt, rows, req = task.batch(nb)
            if oracle:
                set_oracle_dirs(model, req, rows, n, task.T)
            pick = model(x, pay).gather(1, rows.unsqueeze(-1).expand(-1, -1, task.cmax + 1))
            ok += (pick.argmax(-1) == tgt).sum().item(); tot += tgt.numel()
    model.train()
    for b in model.blocks:
        if getattr(b.attn, "ca", None) is not None:
            b.attn.ca.dir_ovr = None
    return ok / tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    for a, t, d in (("--d", int, 3), ("--k", int, 2), ("--T", int, 1024), ("--chunk", int, 256),
                    ("--n-marks", int, 12), ("--n-syms", int, 512), ("--pool", int, 8),
                    ("--n-ans", int, 8), ("--hash-shift", int, 1), ("--hash-conv", int, 0),
                    ("--layers", int, 1), ("--d-model", int, 256), ("--heads", int, 4),
                    ("--r", int, 64), ("--steps", int, 4000), ("--nb", int, 8),
                    ("--eval-every", int, 1000), ("--seed", int, 0), ("--short-conv", int, 1)):
        ap.add_argument(a, type=t, default=d)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arm", choices=["smat", "softmax"], default="smat")
    ap.add_argument("--dirs", choices=["learned", "oracle"], default="oracle")
    ap.add_argument("--out", default="results/counting.csv")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = to_device(build_mask(args.T, max(args.d, 1), chunk=args.chunk), dev)
    vocab = args.n_syms + 3 + 256
    task = Counting(args.T, spec.n, args.n_marks, args.n_syms, args.k, args.n_ans, vocab, dev,
                    seed=args.seed, pool=args.pool)
    name = f"count_{args.arm}_d{args.d}_k{args.k}_T{args.T}_s{args.seed}{args.tag}"
    smat = args.arm == "smat" and args.d >= 2
    print(json.dumps({"run": name, **vars(args), "q": spec.q, "N0": spec.N0, "B": spec.B,
                      "n": spec.n, "cmax": task.cmax, "device": dev,
                      "covers_k": args.k <= max(args.d - 1, 1)}), flush=True)

    model = CountModel(args.T, args.d_model, args.heads, args.layers, task.cmax + 1, vocab,
                       arm=args.arm, short_conv=args.short_conv, spec=spec, r=args.r,
                       chunk=args.chunk, device=dev, seed=args.seed, qk_norm=True,
                       decay="mamba2", dt_init=(0.1, 1.0), A_init=(1.0, 16.0),
                       assign="content" if smat else "positional", read_mode="plane",
                       hash_src="id", hash_freeze=True,
                       hash_shift=(0 if args.hash_conv else args.hash_shift),
                       hash_conv=args.hash_conv).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.DictWriter(f, ["run", "arm", "d", "k", "step", "loss", "acc", "covers_k", "ms_per_step"])
    if new:
        w.writeheader()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, pay, tgt, rows, req = task.batch(args.nb)
        if args.dirs == "oracle" and smat:
            set_oracle_dirs(model, req, rows, spec.n, args.T)
        pick = model(x, pay).gather(1, rows.unsqueeze(-1).expand(-1, -1, task.cmax + 1))
        loss = F.cross_entropy(pick.reshape(-1, task.cmax + 1), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            acc = evaluate(model, task, spec.n, args.nb, 8, args.dirs == "oracle" and smat)
            ms = 1000 * (time.time() - t0) / step
            print(f"== step {step}: loss {loss.item():.4f}  acc {acc:.3f}  {ms:.0f} ms/step", flush=True)
            w.writerow({"run": name, "arm": args.arm, "d": args.d, "k": args.k, "step": step,
                        "loss": loss.item(), "acc": acc,
                        "covers_k": args.k <= max(args.d - 1, 1), "ms_per_step": ms})
            f.flush()
    f.close()


if __name__ == "__main__":
    main()

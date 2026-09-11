"""Multi-key subset recall: retrieval and subset selection in the same task.

N (key, payload) pairs sit at RANDOM positions in the distant block; the payload
of pair i is the basis vector e_i, injected through the value stream.  A query
names k of the keys and must emit a vector whose support is exactly the
requested set.  Both halves of the construction are required:

  * the pairs are placed at random, so a query cannot address them by position
    and the assignment has to be content-addressed;
  * one row must admit exactly the k requested payloads, so its trace on the N
    marked columns has to realise an arbitrary k-subset.

The second is what makes d bite.  In F_q^{d-1} a set of k points lies on a
common hyperplane exactly when k <= d-1 (its affine hull has dimension <= k-1),
so hyperplane types can cover an arbitrary request only when d >= k+1, and point
types -- one cell -- only when k = 1:

    d=2 (hyperplane = point)      k = 1
    d=3 (hyperplane = line)       k <= 2
    d=4 (hyperplane = plane)      k <= 3

Two direction modes.  ``learned`` picks the hyperplane's direction by a learned
map of the query, which is what the model would have to do unaided; ``oracle``
hands it a direction on which all k requested cells agree, when one exists.  The
oracle arm asks whether the MASK suffices; the gap between them is the learning
problem, and without the split a failure says nothing about which.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "smat"))
from smat_mask import build_mask                                   # noqa: E402
from smat_attn import to_device                                    # noqa: E402
import train_lm                                                    # noqa: E402


class MKAR:
    """Sequence: noise everywhere, N (key, payload) pairs at random even slots in
    the distant block, and n_ans answer rows in the recent block, each preceded
    by the k key tokens it requests."""

    def __init__(self, T, n, n_pairs, k, n_ans, vocab, device, seed=0, ablate="none"):
        self.ablate = ablate
        assert 2 * n_pairs + 2 < n and n_ans * (k + 1) < T - n
        self.T, self.n, self.N, self.k, self.n_ans = T, n, n_pairs, k, n_ans
        self.vocab, self.device = vocab, device
        self.rng = np.random.default_rng(seed)
        self.KEY0, self.PAY, self.NOISE0 = 1, 1 + n_pairs * 4, 1 + n_pairs * 4 + 1
        self.n_keys = n_pairs * 4                       # key alphabet, > N so keys vary

    def batch(self, nb):
        T, n, N, k = self.T, self.n, self.N, self.k
        x = self.rng.integers(self.NOISE0, self.vocab, size=(nb, T))
        pay = np.zeros((nb, T, N), dtype=np.float32)
        tgt = np.zeros((nb, self.n_ans, N), dtype=np.float32)
        rows = np.zeros((nb, self.n_ans), dtype=np.int64)
        req = np.zeros((nb, self.n_ans, k), dtype=np.int64)
        slots = np.arange(0, n - 2, 2)
        astep = (T - n) // self.n_ans
        for b in range(nb):
            pos = np.sort(self.rng.choice(slots, N, replace=False))
            keys = self.rng.choice(self.n_keys, N, replace=False)
            x[b, pos] = self.KEY0 + keys
            x[b, pos + 1] = self.PAY                    # payload marker token
            # index of a pair is its position rank; "permidx" breaks that, in case
            # the rank is recoverable from something other than the payload itself
            idx = self.rng.permutation(N) if self.ablate == "permidx" else np.arange(N)
            pay[b, pos + 1, idx] = 1.0
            for m in range(self.n_ans):
                t = n + astep * m + astep - 1           # answer row
                A = self.rng.choice(N, k, replace=False)
                # the answer row itself carries one of the requested keys, and the
                # other k-1 sit in its recent prefix.  The row's TYPE is the hash of
                # its own token, so the hyperplane's offset a^T x(q) is only right if
                # x(q) is one of the cells to be covered; leaving a noise token here
                # points the row at an unrelated offset however good the direction is.
                x[b, t] = self.KEY0 + keys[A[0]]
                if k > 1:
                    x[b, t - (k - 1):t] = self.KEY0 + keys[A[1:]]
                rows[b, m] = t
                tgt[b, m, idx[A]] = 1.0
                req[b, m] = self.KEY0 + keys[A]
        if self.ablate == "shuftgt":        # metric sanity: targets unrelated to the request
            tgt = np.zeros_like(tgt)
            for b in range(nb):
                for m in range(self.n_ans):
                    tgt[b, m, self.rng.choice(N, k, replace=False)] = 1.0
        t = lambda a, d=None: torch.as_tensor(a, device=self.device, dtype=d)
        return (t(x), t(pay, torch.float32), t(tgt, torch.float32),
                t(rows), t(req))


class MKARModel(nn.Module):
    def __init__(self, T, d_model, heads, layers, N, vocab, **kw):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(T, d_model)
        self.pay_in = nn.Linear(N, d_model, bias=False)
        self.blocks = nn.ModuleList([train_lm.Block(d_model, heads, "smat", layer_idx=i, **kw)
                                     for i in range(layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N)

    def forward(self, idx, pay):
        emb = self.tok(idx)
        x = emb + self.pos(torch.arange(idx.shape[1], device=idx.device)) + self.pay_in(pay)
        for b in self.blocks:
            x = b(x, emb=emb)
        return self.head(self.ln_f(x))


@torch.no_grad()
def set_oracle_dirs(model, req, rows, n, T):
    """For every answer row, a direction on which all k requested cells agree."""
    for b in model.blocks:
        ca = getattr(b.attn, "ca", None)
        if ca is None or ca.mode != "plane":
            continue
        e = model.tok(req)                                   # (nb, n_ans, k, d_model)
        cells = ca.cells_of(e)                               # (nb, n_ans, k, h)
        nb, na, k, h = cells.shape
        cells = cells.permute(0, 3, 1, 2).reshape(nb * h, na, k)
        d_ans = ca.oracle_dir(cells)                         # (nb*h, n_ans)
        full = torch.zeros(nb * h, T - n, dtype=torch.long, device=req.device)
        idx = (rows - n).unsqueeze(1).expand(nb, h, na).reshape(nb * h, na)
        full.scatter_(1, idx, d_ans)
        ca.dir_ovr = full


def evaluate(model, task, n, nb, iters, oracle):
    model.eval(); exact = tot = 0; mse = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x, pay, tgt, rows, req = task.batch(nb)
            if oracle:
                set_oracle_dirs(model, req, rows, n, task.T)
            out = model(x, pay)
            pick = out.gather(1, rows.unsqueeze(-1).expand(-1, -1, tgt.shape[-1]))
            mse += F.mse_loss(pick, tgt).item()
            exact += ((pick > 0.5).float() == tgt).all(-1).sum().item()
            tot += tgt.shape[0] * tgt.shape[1]
    model.train()
    for b in model.blocks:
        if getattr(b.attn, "ca", None) is not None:
            b.attn.ca.dir_ovr = None
    return {"exact_support": exact / tot, "mse": mse / iters}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--n-ans", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--nb", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--read-mode", choices=["plane", "point"], default="plane")
    ap.add_argument("--dirs", choices=["learned", "oracle"], default="oracle")
    ap.add_argument("--out", default="results/mkar.csv")
    ap.add_argument("--ablate", choices=["none", "nolr", "permidx", "shuftgt"], default="none",
                    help="leak hunt: nolr disables the long-range block entirely, permidx "
                         "decouples a pair's index from its position rank, shuftgt randomises targets")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec_np = build_mask(args.T, args.d, chunk=args.chunk)
    spec = to_device(spec_np, dev)
    vocab = args.n_pairs * 4 + 2 + 256
    task = MKAR(args.T, spec.n, args.n_pairs, args.k, args.n_ans, vocab, dev, seed=args.seed,
                ablate=args.ablate)
    name = (f"mkar_d{args.d}_k{args.k}_{args.read_mode}_{args.dirs}_T{args.T}"
            f"_s{args.seed}" + ("" if args.ablate == "none" else f"_{args.ablate}") + args.tag)
    print(json.dumps({"run": name, **vars(args), "q": spec.q, "N0": spec.N0, "B": spec.B,
                      "deg": spec.uniform_deg, "n": spec.n, "device": dev,
                      "covers_k": args.k <= max(args.d - 1, 1)}), flush=True)

    model = MKARModel(args.T, args.d_model, args.heads, args.layers, args.n_pairs, vocab,
                      spec=spec, r=args.r, chunk=args.chunk, device=dev, seed=args.seed,
                      qk_norm=True, decay="mamba2", dt_init=(0.1, 1.0), A_init=(1.0, 16.0),
                      assign="content", read_mode=args.read_mode, hash_src="embed",
                      hash_freeze=True, hash_shift=1,
                      no_lr=(args.ablate == "nolr")).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    fields = ["run", "d", "k", "read_mode", "dirs", "step", "loss", "exact_support", "mse",
              "q", "N0", "B", "covers_k", "ms_per_step"]
    f = open(args.out, "a", newline=""); w = csv.DictWriter(f, fields)
    if new:
        w.writeheader()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, pay, tgt, rows, req = task.batch(args.nb)
        if args.dirs == "oracle":
            set_oracle_dirs(model, req, rows, spec.n, args.T)
        out = model(x, pay)
        pick = out.gather(1, rows.unsqueeze(-1).expand(-1, -1, tgt.shape[-1]))
        loss = F.binary_cross_entropy_with_logits(pick, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, task, spec.n, args.nb, 8, args.dirs == "oracle")
            ms = 1000 * (time.time() - t0) / step
            print(f"== step {step}: loss {loss.item():.4f}  exact_support "
                  f"{ev['exact_support']:.3f}  mse {ev['mse']:.4f}  {ms:.0f} ms/step", flush=True)
            w.writerow({"run": name, "d": args.d, "k": args.k, "read_mode": args.read_mode,
                        "dirs": args.dirs, "step": step, "loss": loss.item(),
                        "exact_support": ev["exact_support"], "mse": ev["mse"],
                        "q": spec.q, "N0": spec.N0, "B": spec.B,
                        "covers_k": args.k <= max(args.d - 1, 1), "ms_per_step": ms})
            f.flush()
    f.close()


if __name__ == "__main__":
    main()

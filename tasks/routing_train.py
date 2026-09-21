"""The learned half of d-Subset Routing: does SGD find the routes the mask permits?

``bench_routing.py`` certifies what a mask *permits* -- the ceiling for any model
carrying it.  This trains an actual model against that ceiling.  The mask is
fixed; ``W_Q, W_K, W_V, W_O`` and the positional table are learned.  The question
is whether the transition at ``k = VC(M) = d`` survives optimisation.

Four arms, because the interesting comparison is not SMAT against nothing:

  smat            request conveyed positionally -- the query sits at row t_A.
                  This is how the architecture routes: rho(i) = i mod B is a
                  function of position, so which landmarks a query can reach is
                  decided by where it is, not by what it contains.
  smat_content    the same, plus a k-bit encoding of A in the query embedding.
                  A control: SMAT cannot route on content, so this should not
                  lift the k > d ceiling.  If it does, the mask is not what is
                  doing the work.
  softmax         causal softmax attention, request conveyed positionally.
                  Mask L_T has VC 1, and with no content signal it cannot tell
                  requested landmarks from unrequested ones.
  softmax_content the strong baseline: full causal attention that can content-
                  address the requested landmarks.  Expected to solve every k,
                  approximately and at Theta(T^2).  Running it is the point --
                  a handicapped baseline would make the whole comparison void.
  mamba2          scalar-decay linear recurrence (Mamba-2 / SSD).
  deltanet        the delta rule.
  gated_deltanet  the delta rule with Mamba-2's decay, i.e. Gated DeltaNet.
  loglinear       log-linear attention: linear attention under the hierarchical
                  Fenwick mask, Theta(log T) states rather than one.

The last four are the models the paper claims are pinned at VC 1.  They differ
in how much they carry -- one state, one orthogonalising state, one gated state,
log T states -- and not at all in what their row supports are, which is the
prediction being tested.  ``src/smat/mixers/recurrent.py`` holds all four.

What separates SMAT from softmax_content is not that softmax cannot do the task.
It is *leakage*: unrequested payloads are structurally absent from SMAT's output
(the mask removed the edges) and merely small in softmax's.  Both are reported.

Training must use the torch backends.  The Triton kernels are raw ``@triton.jit``
functions, not ``autograd.Function``s, so on CUDA with ``backend="auto"`` they
would be selected and gradients would be wrong or missing *silently*.
``_assert_differentiable`` guards this at startup.

Usage
-----
    python train_routing.py --pilot                          # one config, ~5 min
    python train_routing.py --T 1024 --d 1 2 3 4 5 --kmax 6 --seeds 3 --plot
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from smat.mask import build_mask, d_max_geometric
from smat.attention import featurise, make_phi, smat_attention, to_device
from routing_ceiling import (choose_marked_set, pattern_table, row_support_size,
                           shatter_profile)
from smat.mixers.recurrent import (Mamba2Gate, DeltaGate, GatedDeltaGate,
                            LogLinearGate, mamba2_chunked, delta_chunked,
                            gated_delta_chunked, loglinear_attention,
                            fenwick_levels)

ARMS = ["smat", "smat_content", "softmax", "softmax_content",
        "mamba2", "deltanet", "gated_deltanet", "loglinear"]


def _wandb_init(enabled: bool, project: str, cfg: Dict):
    """Optional, off by default.

    The sweep's evidence is the CSV -- 126 runs of a few seconds each, whose
    final metrics are what the figure reads, and whose threat to validity (is a
    bad number a ceiling or a failed fit?) is answered by the certified ceiling
    and the loss_q* columns rather than by a dashboard.  This is here for
    interactive inspection of a single run.  Authenticate with `wandb login` or
    WANDB_API_KEY in the environment; never put a key in this file.
    """
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("    --wandb given but wandb is not installed; continuing without")
        return None
    if not (os.environ.get("WANDB_API_KEY") or
            os.path.exists(os.path.expanduser("~/.netrc"))):
        print("    --wandb given but no credential found "
              "(set WANDB_API_KEY or run `wandb login`); continuing without")
        return None
    try:
        return wandb.init(project=project, config=cfg, reinit=True,
                          mode=os.environ.get("WANDB_MODE", "online"))
    except Exception as e:                      # a batch node with no egress
        print(f"    wandb init failed ({e}); continuing without")
        return None


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class RoutingTask:
    """Sec. 4's task: payloads at marked positions, a request A, target y_A.

    Payloads are resampled every batch, so nothing can be memorised; the ``2^k``
    requests are all seen by construction (there are only ``2^k`` of them), which
    is what the draft intends -- generalisation is over payloads.
    """

    def __init__(self, spec, cols: Sequence[int], *, device, rng):
        self.spec, self.cols, self.device, self.rng = spec, list(cols), device, rng
        self.k = len(cols)
        self.T = spec.T
        self.pat = pattern_table(spec, cols)
        self.nu = torch.as_tensor(row_support_size(spec), device=device,
                                  dtype=torch.float32)
        # the row serving each request, and the best available row when none does
        self.row_of = np.array([self._row(A) for A in range(1 << self.k)])
        self.answerable = np.array([A in self.pat for A in range(1 << self.k)])

    def _row(self, A: int) -> int:
        if A in self.pat:
            return self.pat[A]
        best, bd = None, self.k + 1
        for p, t in self.pat.items():
            dist = int(p ^ A).bit_count()
            if dist < bd:
                best, bd = t, dist
        return best

    def batch(self, nb: int, rows_mode: str = "mask"):
        """``(payload, rows, A_bits, y, answerable, x)``.

        ``rows_mode`` picks where the query sits.  ``"mask"`` uses the row whose
        mask pattern realises A -- correct for SMAT, whose routing is positional.
        ``"last"`` puts the query after every marked position, which is what a
        content-addressed baseline needs: inheriting SMAT's row would park the
        softmax query before the landmarks it is supposed to choose between,
        where causality alone makes them invisible.  That is not a baseline, it
        is a handicap.

        ``x`` is the raw payload vector.  Metrics are normalised by it rather
        than by ``y_A``: the empty request has ``y_A = 0``, so any quantity
        divided by ``||y_A||`` is undefined on one request in ``2^k`` -- and
        that request is exactly the one testing whether the model correctly
        emits nothing.
        """
        k = self.k
        A = self.rng.integers(0, 1 << k, size=nb)
        x = self.rng.standard_normal((nb, k)).astype(np.float32)
        bits = ((A[:, None] >> np.arange(k)[None, :]) & 1).astype(np.float32)

        pay = np.zeros((nb, self.T, k), dtype=np.float32)
        for l, j in enumerate(self.cols):
            pay[:, int(j), l] = x[:, l]
        y = x * bits

        if rows_mode == "last":
            rows = np.full(nb, self.T - 1, dtype=np.int64)
        else:
            rows = self.row_of[A]

        dev = self.device
        return (torch.as_tensor(pay, device=dev),
                torch.as_tensor(rows, device=dev, dtype=torch.long),
                torch.as_tensor(bits, device=dev),
                torch.as_tensor(y, device=dev),
                torch.as_tensor(self.answerable[A], device=dev),
                torch.as_tensor(x, device=dev))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class Router(nn.Module):
    """One attention layer over the marked context, with a linear readout.

    The payload enters through the value stream, position through a learned
    table, and (in the ``*_content`` arms) the request through a projection
    added at the query row only.  Undoing attention's mean is a fixed,
    mask-determined rescale by ``|S_t|``; everything else is learned.
    """

    def __init__(self, T: int, k: int, *, arm: str, d_model: int = 64,
                 d_qk: int = 32, d_v: int = 32, r: int = 32, spec=None,
                 device=None, seed: int = 0):
        super().__init__()
        self.arm, self.k, self.spec = arm, k, spec
        self.use_content = arm.endswith("_content")
        self.is_smat = arm.startswith("smat")
        self.chunk = 128

        self.pos = nn.Embedding(T, d_model)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.pay_in = nn.Linear(k, d_model, bias=False)
        self.req_in = nn.Linear(k, d_model, bias=False) if self.use_content else None

        self.Wq = nn.Linear(d_model, d_qk, bias=False)
        self.Wk = nn.Linear(d_model, d_qk, bias=False)
        self.Wv = nn.Linear(d_model, d_v, bias=False)
        self.Wo = nn.Linear(d_v, k, bias=False)

        if self.is_smat:
            self.phi = make_phi(d_qk, r, device=device, dtype=torch.float32,
                                seed=seed)
        # the recurrent baselines: a causal-prefix row support and VC 1
        # whatever the gate does, and whether the state is one matrix (mamba2,
        # deltanet, gated_deltanet) or log T of them (loglinear)
        self.gate = (Mamba2Gate(d_model) if arm == "mamba2" else
                     DeltaGate(d_model) if arm == "deltanet" else
                     GatedDeltaGate(d_model) if arm == "gated_deltanet" else
                     LogLinearGate(d_model, T) if arm == "loglinear" else None)
        if arm == "loglinear":
            # position-only and fixed, so a buffer rather than a parameter
            self.register_buffer("lev", fenwick_levels(T), persistent=False)

    def forward(self, pay, rows, bits, nu):
        nb, T, _ = pay.shape
        idx = torch.arange(T, device=pay.device)
        X = self.pos(idx).unsqueeze(0) + self.pay_in(pay)
        if self.use_content:
            add = self.req_in(bits)                              # (nb, d_model)
            X = X.index_put((torch.arange(nb, device=pay.device), rows),
                            X[torch.arange(nb, device=pay.device), rows] + add)

        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)

        if self.is_smat:
            Phi, Psi, Vb = featurise(Q, K, V, self.phi)
            o = smat_attention(Phi, Psi, Vb, self.spec, chunk=128,
                               acc_dtype=torch.float32,
                               scan_backend="torch", incidence_backend="torch")
            o = o * nu.view(1, T, 1)                 # undo the mask-determined mean
        elif self.arm == "mamba2":
            logA, dt = self.gate(X)
            o = mamba2_chunked(Q, K, V, logA, dt, chunk=self.chunk)
        elif self.arm == "deltanet":
            o = delta_chunked(Q, K, V, self.gate(X), chunk=self.chunk)
        elif self.arm == "gated_deltanet":
            logA, beta = self.gate(X)
            o = gated_delta_chunked(Q, K, V, logA, beta, chunk=self.chunk)
        elif self.arm == "loglinear":
            o = loglinear_attention(Q, K, V, self.gate(X), self.lev,
                                    chunk=self.chunk)
        else:
            o = F.scaled_dot_product_attention(
                Q.unsqueeze(1), K.unsqueeze(1), V.unsqueeze(1), is_causal=True
            ).squeeze(1)

        got = o[torch.arange(nb, device=pay.device), rows]        # (nb, d_v)
        return self.Wo(got)


def _assert_differentiable(spec, device):
    """Gradients must reach the projections; the Triton path would break this."""
    phi = make_phi(8, 8, device=device, dtype=torch.float32, seed=0)
    W = torch.randn(4, 8, device=device, requires_grad=True)
    X = torch.randn(1, spec.T, 4, device=device)
    Q = K = X @ W
    V = X @ torch.randn(4, 4, device=device)
    Phi, Psi, Vb = featurise(Q, K, V, phi)
    o = smat_attention(Phi, Psi, Vb, spec, chunk=128, acc_dtype=torch.float32,
                       scan_backend="torch", incidence_backend="torch")
    o.pow(2).sum().backward()
    if W.grad is None or not torch.isfinite(W.grad).all():
        raise RuntimeError("no finite gradient through smat_attention -- refusing "
                           "to train (check the backend is 'torch', not triton)")


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def train_one(spec, cols, arm: str, *, steps: int, nb: int, lr: float,
              device, seed: int, eval_nb: int = 512,
              log_every: int = 0, wandb_run=None) -> Dict[str, object]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    task = RoutingTask(spec, cols, device=device, rng=rng)
    k = len(cols)

    # a content-addressed baseline must be able to see what it is choosing between
    rows_mode = "last" if arm == "softmax_content" else "mask"
    model = Router(spec.T, k, arm=arm, spec=spec, device=device,
                   seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    hist: List[float] = []
    t0 = time.perf_counter()
    for s in range(steps):
        pay, rows, bits, y, _, _ = task.batch(nb, rows_mode)
        pred = model(pay, rows, bits, task.nu)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
        if wandb_run is not None:
            wandb_run.log({"loss": hist[-1], "step": s})
        if log_every and (s % log_every == 0 or s == steps - 1):
            print(f"      step {s:5d}  loss {loss.item():.3e}", flush=True)
    train_s = time.perf_counter() - t0
    # Enough of the loss curve to tell a real ceiling from a failed fit, without
    # a logging service: a k <= d run that does not converge is an optimisation
    # artefact and has to be visible as one.
    q = [hist[min(len(hist) - 1, int(f * len(hist)))] for f in (0.25, .5, .75)]
    tail = hist[-max(10, len(hist) // 10):]
    conv = dict(loss_final=hist[-1], loss_best=min(hist),
                loss_q25=q[0], loss_q50=q[1], loss_q75=q[2],
                loss_tail_slope=float(np.polyfit(
                    np.arange(len(tail)), np.log(np.maximum(tail, 1e-12)), 1)[0]))

    # -- evaluation on fresh payloads -------------------------------------
    model.eval()
    with torch.no_grad():
        pay, rows, bits, y, ans, x = task.batch(eval_nb, rows_mode)
        # Causal influence of the UNREQUESTED payloads: resample them and see
        # whether the prediction moves at all.  This is the sharp form of "un-
        # requested items are exactly non-influential because the mask removes
        # their edges" -- for SMAT at k <= d it must be bit-exactly zero, and no
        # amount of training error can fake that.  A content-addressed baseline
        # attends to those positions with small positive weight, so its
        # influence is small but nonzero.  Unlike leakage, this isolates the
        # architecture from the quality of the fit.
        pay2 = pay.clone()
        x2 = torch.as_tensor(task.rng.standard_normal(tuple(x.shape))
                             .astype(np.float32), device=x.device)
        off = 1.0 - bits
        for l, j in enumerate(task.cols):
            pay2[:, int(j), l] = x[:, l] * bits[:, l] + x2[:, l] * off[:, l]
        pred = model(pay, rows, bits, task.nu)
        # normalise by the payload energy present, not by ||y_A||: the latter is
        # zero for the empty request.
        den = (x ** 2).sum(-1).clamp_min(1e-12)
        nmse = ((pred - y) ** 2).sum(-1) / den

        on = bits.bool()
        leak = pred.masked_fill(on, 0.0).abs().amax(-1)          # unrequested mass
        sig = x.abs().amax(-1).clamp_min(1e-12)                  # payload scale
        # a channel counts as "on" if it carries >1% of the payload scale
        got_on = (pred.abs() > 0.01 * sig.unsqueeze(-1))
        exact = (got_on == on).all(-1).float()

        pred2 = model(pay2, rows, bits, task.nu)
        influence = (pred2 - pred).abs().amax(-1) / sig

    a = ans.bool()
    return dict(
        arm=arm, k=k, seed=seed, steps=steps, rows_mode=rows_mode, **conv,
        nmse=float(nmse.mean()),
        nmse_answerable=float(nmse[a].mean()) if a.any() else float("nan"),
        nmse_unanswerable=float(nmse[~a].mean()) if (~a).any() else float("nan"),
        exact_support_frac=float(exact.mean()),
        leak_rel=float((leak / sig).mean()),
        leak_max=float((leak / sig).amax()),
        influence_mean=float(influence.mean()),
        influence_max=float(influence.amax()),
        influence_zero_frac=float((influence == 0).float().mean()),
        train_seconds=train_s)


def _wandb_finish(run, metrics: Dict):
    if run is not None:
        run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        run.finish()


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def sweep(args, device) -> List[Dict]:
    rows: List[Dict] = []
    rng = np.random.default_rng(args.seed)
    ks = list(range(1, args.kmax + 1))

    for d in args.d:
        dmax = d_max_geometric(args.T, chunk=args.chunk)
        if d > dmax:
            print(f"d={d}: skipped (above d_max({args.T})={dmax})")
            continue
        spec_np = build_mask(args.T, d, chunk=args.chunk)
        spec = to_device(spec_np, device)
        _assert_differentiable(spec, device)
        print(f"\n{spec_np.summary()}")

        for k in ks:
            cols, how = choose_marked_set(spec_np, k, rng=rng,
                                          candidates=args.candidates)
            prof = shatter_profile(spec_np, cols)
            arms = [a for a in args.arms if a.startswith("smat")]
            # the baselines have no mask parameter; run each once, under
            # d = the first d, on that d's marked set
            if d == args.d[0]:
                arms += [a for a in args.arms if not a.startswith("smat")]
            for arm in arms:
                for seed in range(args.seeds):
                    run = _wandb_init(args.wandb, args.wandb_project,
                                      dict(T=args.T, d=d, k=k, arm=arm,
                                           seed=seed, steps=args.steps,
                                           nb=args.nb, lr=args.lr))
                    r = train_one(spec, cols, arm, steps=args.steps, nb=args.nb,
                                  lr=args.lr, device=device, seed=seed,
                                  log_every=args.log_every, wandb_run=run)
                    _wandb_finish(run, r)
                    r.update(T=args.T, d=(d if arm.startswith("smat") else 0),
                             cols=" ".join(map(str, cols)), source=how,
                             ceiling_answerable=prof["answerable_frac"],
                             n_patterns=prof["n_patterns"])
                    rows.append(r)
                    print(f"  d={r['d']} k={k} {arm:16s} seed{seed}  "
                          f"nmse {r['nmse']:.3e}  supp {r['exact_support_frac']:.3f}"
                          f"  infl {r['influence_max']:.2e}"
                          f"  zero {r['influence_zero_frac']:.2f}  ceiling "
                          f"{prof['answerable_frac']:.3f}  "
                          f"({r['train_seconds']:.0f}s)", flush=True)
    return rows


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
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
    print(f"\nwrote {path} ({len(rows)} rows)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--d", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--nb", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--candidates", type=int, default=96)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/routing_learned.csv")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pilot", action="store_true",
                    help="one d=3 config, verbose, to check convergence")
    ap.add_argument("--wandb", action="store_true",
                    help="optional per-run logging; off by default (the CSV is "
                         "the artefact).  Needs WANDB_API_KEY or `wandb login`.")
    ap.add_argument("--wandb-project", default="smat-routing")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args(argv)

    if args.pilot:
        args.d, args.kmax, args.seeds = [3], 4, 1
        args.log_every = args.log_every or 250
        args.out = "results/routing_learned_pilot.csv"

    device = torch.device(args.device)
    print(f"device: {device}   torch {torch.__version__}   "
          f"threads {torch.get_num_threads()}")
    t0 = time.perf_counter()
    rows = sweep(args, device)
    write_csv(rows, args.out)
    print(f"total {(time.perf_counter() - t0) / 60:.1f} min")

    if args.plot:
        import plot_smat
        P = plot_smat.theme(args.dark)
        plot_smat.apply_style(P)
        os.makedirs(args.outdir, exist_ok=True)
        ceiling = os.path.join(os.path.dirname(args.out) or ".",
                               f"routing_T{args.T}.csv")
        rows_c = plot_smat.load(ceiling) if os.path.exists(ceiling) else None
        plot_smat.fig_routing_learned(rows, P, args.outdir, rows_c)
    return 0


if __name__ == "__main__":
    sys.exit(main())

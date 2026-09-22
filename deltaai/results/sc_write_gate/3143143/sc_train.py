"""Selective Copying (Mamba paper, Sec. 4.1.1): L = 4096, 16 content tokens (vocab 16) at random positions among
noise tokens, then 16 answer slots (blank input) whose targets are the content tokens in order.  Loss / accuracy on
the 16 answer slots.  Same model as lm/train_lm2.py (embedding -> n x [RMSNorm -> SmatMamba2MR] -> RMSNorm -> tied head);
arms differ only in the mixer's d / reset.  Resumable (--ckpt)."""
import argparse, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/u/archerdw/smat_attention/deltaai")
import zoo_smat_mixer as z

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="mamba2", choices=["mamba2", "smat", "gdn"]); ap.add_argument("--d", type=int, default=3)
ap.add_argument("--reset", type=int, default=1)
ap.add_argument("--d_model", type=int, default=64); ap.add_argument("--n_layers", type=int, default=2)
ap.add_argument("--d_state", type=int, default=64); ap.add_argument("--headdim", type=int, default=64)
ap.add_argument("--seq_len", type=int, default=4096); ap.add_argument("--n_copy", type=int, default=16); ap.add_argument("--n_vocab", type=int, default=16)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--steps", type=int, default=400_000)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--ckpt", required=True); ap.add_argument("--ckpt_every", type=int, default=500)
ap.add_argument("--eval_every", type=int, default=500); ap.add_argument("--log_every", type=int, default=50)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max_minutes", type=float, default=110)
ap.add_argument("--hash_conv_width", type=int, default=4); ap.add_argument("--anneal", type=int, default=2000)
ap.add_argument("--balance", type=float, default=0.01); ap.add_argument("--balance_gated", type=int, default=1)
ap.add_argument("--hash_lr", type=float, default=1.0); ap.add_argument("--hash_codim", type=int, default=1)   # 1 = hyperplane (paper), d-1 = point
ap.add_argument("--stop_acc", type=float, default=0.995)
ap.add_argument("--lam_act", default="one", choices=["one", "sigmoid", "zero"]); ap.add_argument("--hash_src", default="hidden", choices=["hidden", "embed", "ssm"])   # sigmoid: learned per-head gate on the G read (init off); ssm: hash the SSM output
ap.add_argument("--gdn_headdim", type=int, default=None)   # GDN head_dim (state per head = head_dim x 2*head_dim); default = --headdim   # stop once two consecutive evals reach this accuracy
ap.add_argument("--g_write_mode", choices=["shared", "independent"], default="shared",
                help="G writes use Mamba-2 dt (shared) or a separate per-head sigmoid gate (independent)")
ap.add_argument("--g_write_init", type=float, default=0.1, help="initial independent G write probability")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed); dev = "cuda"
NOISE, BLANK, VOCAB = args.n_vocab, args.n_vocab + 1, args.n_vocab + 2      # 0..15 content, 16 noise, 17 blank


def make_batch(b, gen=None):
    L, k = args.seq_len, args.n_copy
    x = torch.full((b, L), NOISE, dtype=torch.long)
    content = torch.randint(0, args.n_vocab, (b, k), generator=gen)
    for i in range(b):                                                     # k distinct random positions in the prefix, sorted
        pos = torch.randperm(L - k, generator=gen)[:k].sort().values
        x[i, pos] = content[i]
    x[:, L - k:] = BLANK
    y = torch.full((b, L), -100, dtype=torch.long); y[:, L - k:] = content
    return x, y


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w


class GDN(nn.Module):
    """Gated DeltaNet baseline (flash-linear-attention layer as is: chunk mode, gate, short conv, expand_v 2)."""
    def __init__(self, i):
        super().__init__()
        from fla.layers.gated_deltanet import GatedDeltaNet
        hd = args.gdn_headdim or args.headdim
        self.gdn = GatedDeltaNet(hidden_size=args.d_model, expand_v=2, head_dim=hd, num_heads=max(1, args.d_model // hd),
                                 mode="chunk", use_gate=True, use_short_conv=True, layer_idx=i)
    def forward(self, x): return self.gdn(x)[0]
    def get_auxiliary_loss(self): return 0.0


class Block(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.norm = RMSNorm(args.d_model)
        if args.arm == "gdn":
            self.mixer = GDN(i); return
        d = 1 if (args.arm == "mamba2" or args.lam_act == "zero") else args.d      # "zero": G switched off entirely = the bare recurrence
        self.mixer = z.SmatMamba2MR(args.d_model, layer_idx=i, d=d, d_state=args.d_state, headdim=args.headdim,
                                    lam_act="one" if args.lam_act == "zero" else args.lam_act, hash_src=args.hash_src, reset=bool(args.reset), g_decay=True, read="chash", hash_mode="point",
                                    hash_codim=args.hash_codim, hash_conv=True, hash_conv_width=args.hash_conv_width,
                                    anneal_steps=args.anneal, balance_coef=args.balance, balance_gated=bool(args.balance_gated),
                                    hash_lr_scale=args.hash_lr, hash_ckpt=True, sparse_ops=True, g_bf16=True,
                                    g_write_mode=args.g_write_mode, g_write_init=args.g_write_init)
    def forward(self, x):
        return x + self.mixer(self.norm(x))


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, args.d_model); self.blocks = nn.ModuleList([Block(i) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.d_model); nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, idx):
        x = self.emb(idx); z._EMB["x"] = x.detach()
        for b in self.blocks: x = b(x)
        return F.linear(self.norm(x), self.emb.weight)
    def aux(self):
        return sum(b.mixer.get_auxiliary_loss() for b in self.blocks)


def lr_at(step):
    if step < args.warmup: return args.lr * step / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


model = LM().to(dev)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):      # build the lazy hash modules BEFORE the optimizer (else the hash never trains)
    model(torch.zeros(1, args.seq_len, dtype=torch.long, device=dev))
model.train()
print(f"arm={args.arm} d={args.d} reset={args.reset} layers={args.n_layers} d_model={args.d_model} params={sum(p.numel() for p in model.parameters())} L={args.seq_len} batch={args.batch}", flush=True)
print(f"G write mode={args.g_write_mode} init={args.g_write_init} read={args.lam_act} hash={args.hash_src} seed={args.seed}", flush=True)
decay, no_decay = [], []
for n_, p in model.named_parameters():
    (no_decay if p.ndim < 2 or "A_log" in n_ or "dt_bias" in n_ or n_.endswith(".D") else decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
n_opt = sum(len(g["params"]) for g in opt.param_groups); n_model = sum(1 for _ in model.parameters()); assert n_opt == n_model, (n_opt, n_model)
print(f"optimizer covers {n_opt}/{n_model} parameter tensors ({sum(1 for n_, _ in model.named_parameters() if '.ca.' in n_)} content-hash tensors)", flush=True)
step = 0
if os.path.exists(args.ckpt):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): model(torch.zeros(1, args.seq_len, dtype=torch.long, device=dev))
    ck = torch.load(args.ckpt, map_location=dev)
    if ck["args"].get("g_write_mode", "shared") != args.g_write_mode:
        raise ValueError("checkpoint G write mode does not match this run; use a new checkpoint path")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    for b in model.blocks: b.mixer._steps = step
    print(f"resumed at step {step}", flush=True)
eval_gen = torch.Generator().manual_seed(1234); xe, ye = make_batch(256, eval_gen); xe, ye = xe.to(dev), ye.to(dev)


def evaluate():
    model.eval(); correct = tot = 0
    slot_hits = torch.zeros(args.n_copy, device=dev)
    half_hits = torch.zeros(2, device=dev); half_counts = torch.zeros(2, device=dev)
    gate_sums = torch.zeros(args.n_layers, 2, device=dev); gate_counts = torch.zeros_like(gate_sums)
    for block in model.blocks:
        block.mixer.track_g_write_stats = True
    with torch.no_grad():
        for i in range(0, 256, 64):
            with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(xe[i:i+64])
            m = ye[i:i+64] != -100
            correct += (lg.argmax(-1)[m] == ye[i:i+64][m]).sum().item(); tot += m.sum().item()
            hits_by_slot = lg[:, -args.n_copy:].argmax(-1) == ye[i:i+64, -args.n_copy:]
            slot_hits += hits_by_slot.sum(0)
            prefix = xe[i:i+64, :-args.n_copy]
            positions = (prefix < args.n_vocab).nonzero()[:, 1].reshape(prefix.shape[0], args.n_copy)
            for half in range(2):
                belongs = (positions < args.seq_len // 2) if half == 0 else (positions >= args.seq_len // 2)
                half_hits[half] += (hits_by_slot & belongs).sum(); half_counts[half] += belongs.sum()
            for layer, block in enumerate(model.blocks):
                writes = getattr(block.mixer, "last_g_write_weights", None)
                if writes is None:
                    continue
                writes = writes[:, :args.seq_len // 2]
                tokens = xe[i:i+64, :args.seq_len // 2]
                for kind, select in enumerate((tokens < args.n_vocab, tokens == NOISE)):
                    gate_sums[layer, kind] += (writes * select.unsqueeze(-1)).sum()
                    gate_counts[layer, kind] += select.sum() * writes.shape[-1]
    for block in model.blocks:
        block.mixer.track_g_write_stats = False
        if hasattr(block.mixer, "last_g_write_weights"):
            del block.mixer.last_g_write_weights
    print("SLOTS " + " ".join(f"{a:.4f}" for a in (slot_hits / 256).tolist()), flush=True)
    halves = (half_hits / half_counts.clamp_min(1)).tolist()
    print(f"SOURCE_HALF first={halves[0]:.4f} second={halves[1]:.4f}", flush=True)
    if gate_counts.sum() > 0:
        for layer, (content, noise) in enumerate((gate_sums / gate_counts.clamp_min(1)).tolist()):
            print(f"G_WRITE layer={layer} content={content:.6f} noise={noise:.6f}", flush=True)
    model.train(); return correct / tot


t0 = time.time(); losses = []; hits = 0
while step < args.steps:
    xb, yb = make_batch(args.batch); xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
    for g in opt.param_groups: g["lr"] = lr_at(step)
    with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(xb)
    loss = F.cross_entropy(lg.float().view(-1, VOCAB), yb.view(-1), ignore_index=-100)
    (loss + model.aux()).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True); step += 1; losses.append(loss.item())
    if step % args.log_every == 0:
        print(f"step {step} loss {np.mean(losses[-args.log_every:]):.4f} lr {lr_at(step):.1e} {step*args.batch*args.seq_len/(time.time()-t0)/1e3:.0f} ktok/s(this job) {(time.time()-t0)/60:.1f}m", flush=True)
    if step % args.eval_every == 0:
        acc = evaluate(); print(f"EVAL step {step} acc {acc:.4f} peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
        hits = hits + 1 if acc >= args.stop_acc else 0
        if hits >= 2:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, args.ckpt)
            print(f"SOLVED at step {step} (acc {acc:.4f} twice); checkpointed", flush=True); break
    if step % args.ckpt_every == 0 or step == args.steps:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, args.ckpt)
    if (time.time() - t0) / 60 > args.max_minutes:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, args.ckpt)
        print(f"TIME LIMIT at step {step}; checkpointed", flush=True); break
print(f"FINAL step {step} acc {evaluate():.4f}", flush=True)

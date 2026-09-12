"""Selective Copying (Mamba paper, Sec. 4.1.1): L = 4096, 16 content tokens (vocab 16) at random positions among
noise tokens, then 16 answer slots (blank input) whose targets are the content tokens in order.  Loss / accuracy on
the 16 answer slots.  Same model as lm/train_lm2.py (embedding -> n x [RMSNorm -> SmatMamba2MR] -> RMSNorm -> tied head);
arms differ only in the mixer's d / reset.  Resumable (--ckpt)."""
import argparse, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/u/archerdw/smat_attention/deltaai")
import zoo_smat_mixer as z

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="mamba2", choices=["mamba2", "smat"]); ap.add_argument("--d", type=int, default=3)
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
ap.add_argument("--hash_lr", type=float, default=1.0)
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


class Block(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.norm = RMSNorm(args.d_model)
        d = 1 if args.arm == "mamba2" else args.d
        self.mixer = z.SmatMamba2MR(args.d_model, layer_idx=i, d=d, d_state=args.d_state, headdim=args.headdim,
                                    lam_act="one", reset=bool(args.reset), g_decay=True, read="chash", hash_mode="point",
                                    hash_codim=d - 1, hash_conv=True, hash_conv_width=args.hash_conv_width,
                                    anneal_steps=args.anneal, balance_coef=args.balance, balance_gated=bool(args.balance_gated),
                                    hash_lr_scale=args.hash_lr, hash_ckpt=True)
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
print(f"arm={args.arm} d={args.d} reset={args.reset} layers={args.n_layers} d_model={args.d_model} params={sum(p.numel() for p in model.parameters())} L={args.seq_len} batch={args.batch}", flush=True)
decay, no_decay = [], []
for n_, p in model.named_parameters():
    (no_decay if p.ndim < 2 or "A_log" in n_ or "dt_bias" in n_ or n_.endswith(".D") else decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
step = 0
if os.path.exists(args.ckpt):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): model(torch.zeros(1, args.seq_len, dtype=torch.long, device=dev))
    ck = torch.load(args.ckpt, map_location=dev); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    for b in model.blocks: b.mixer._steps = step
    print(f"resumed at step {step}", flush=True)
eval_gen = torch.Generator().manual_seed(1234); xe, ye = make_batch(256, eval_gen); xe, ye = xe.to(dev), ye.to(dev)


def evaluate():
    model.eval(); correct = tot = 0
    with torch.no_grad():
        for i in range(0, 256, 64):
            with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(xe[i:i+64])
            m = ye[i:i+64] != -100
            correct += (lg.argmax(-1)[m] == ye[i:i+64][m]).sum().item(); tot += m.sum().item()
    model.train(); return correct / tot


t0 = time.time(); losses = []
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
        print(f"EVAL step {step} acc {evaluate():.4f}", flush=True)
    if step % args.ckpt_every == 0 or step == args.steps:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, args.ckpt)
    if (time.time() - t0) / 60 > args.max_minutes:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, args.ckpt)
        print(f"TIME LIMIT at step {step}; checkpointed", flush=True); break
print(f"FINAL step {step} acc {evaluate():.4f}", flush=True)

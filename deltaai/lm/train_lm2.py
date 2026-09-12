"""Small language-model trainer for the per-position-loss study.

Model: token embedding (tied head) -> n_layers x [RMSNorm -> mixer] -> RMSNorm -> logits.
Mixer: zoo_smat_mixer.SmatMamba2MR for every arm, so the arms differ only in G:
    --arm mamba2       d=1, lambda off              (plain Mamba-2)
    --arm smat         content-hashed SMAT G on top (paper mask, reset at T/2, v10 recipe by default)
Data: uint16 GPT-2 token memmaps (lm/prep_pg19.py); random windows of --seq_len+1 tokens.
Checkpoints every --ckpt_every steps to --ckpt (resumable: the job limit is 2 h)."""
import argparse, math, os, sys, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/u/archerdw/smat_attention/deltaai")
import zoo_smat_mixer as z

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="mamba2", choices=["mamba2", "smat"])
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--d_model", type=int, default=512); ap.add_argument("--n_layers", type=int, default=8)
ap.add_argument("--d_state", type=int, default=64); ap.add_argument("--headdim", type=int, default=64)
ap.add_argument("--seq_len", type=int, default=16384); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--data", default="/work/hdd/bekw/archerdw/pg19"); ap.add_argument("--vocab", type=int, default=50257)
ap.add_argument("--ckpt", required=True); ap.add_argument("--ckpt_every", type=int, default=250)
ap.add_argument("--val_every", type=int, default=250); ap.add_argument("--log_every", type=int, default=10)
ap.add_argument("--synthetic", action="store_true", help="random tokens (smoke / benchmark)")
ap.add_argument("--seed", type=int, default=0)
# SMAT / hash recipe (v10 defaults)
ap.add_argument("--hash_codim", type=int, default=None); ap.add_argument("--hash_conv", type=int, default=1)
ap.add_argument("--anneal", type=int, default=1000); ap.add_argument("--balance", type=float, default=0.01)
ap.add_argument("--balance_gated", type=int, default=1); ap.add_argument("--hash_lr", type=float, default=1.0)
ap.add_argument("--hash_src", default="hidden"); ap.add_argument("--hash_freeze", type=int, default=0)
ap.add_argument("--hash_conv_width", type=int, default=16); ap.add_argument("--reset", type=int, default=1)
ap.add_argument("--bytes", default="", help="raw byte file (enwik8): 90M train / 5M val / 5M test; sets vocab 256")
args = ap.parse_args()
if args.bytes: args.vocab = 256
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda"


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
        codim = args.hash_codim if args.hash_codim is not None else (d - 1)   # point read
        self.mixer = z.SmatMamba2MR(args.d_model, layer_idx=i, d=d, d_state=args.d_state, headdim=args.headdim,
                                    lam_act="one", reset=bool(args.reset), g_decay=True, read="chash", hash_mode="point",
                                    hash_codim=codim, hash_conv=bool(args.hash_conv), anneal_steps=args.anneal,
                                    balance_coef=args.balance, balance_gated=bool(args.balance_gated),
                                    hash_lr_scale=args.hash_lr, hash_src=args.hash_src, hash_freeze=bool(args.hash_freeze), hash_conv_width=args.hash_conv_width, hash_ckpt=True)
    def forward(self, x):
        return x + self.mixer(self.norm(x))


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(args.vocab, args.d_model)
        self.blocks = nn.ModuleList([Block(i) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, idx):
        x = self.emb(idx)
        z._EMB["x"] = x.detach()
        for b in self.blocks: x = b(x)
        return F.linear(self.norm(x), self.emb.weight)              # tied head
    def aux(self):
        return sum(b.mixer.get_auxiliary_loss() for b in self.blocks)


def load_split(split):
    if args.bytes:
        raw = np.memmap(args.bytes, dtype=np.uint8, mode="r")
        lo, hi = {"train": (0, 90_000_000), "val": (90_000_000, 95_000_000), "test": (95_000_000, 100_000_000)}[split]
        return raw[lo:hi]
    return np.memmap(f"{args.data}/{split}.bin", dtype=np.uint16, mode="r")


def batches(split, n):
    if args.synthetic:
        while True: yield torch.randint(0, args.vocab, (args.batch, n + 1))
    data = load_split(split)
    L = len(data)
    while True:
        ix = np.random.randint(0, L - n - 1, size=args.batch)
        yield torch.stack([torch.from_numpy(data[i:i + n + 1].astype(np.int64)) for i in ix])


def lr_at(step):
    if step < args.warmup: return args.lr * step / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


model = LM().to(dev)
n_params = sum(p.numel() for p in model.parameters()); n_emb = model.emb.weight.numel()
print(f"arm={args.arm} d={args.d} layers={args.n_layers} d_model={args.d_model} params={n_params/1e6:.1f}M (non-emb {(n_params-n_emb)/1e6:.1f}M) T={args.seq_len} batch={args.batch}", flush=True)
decay, no_decay = [], []
for n_, p in model.named_parameters():
    (no_decay if p.ndim < 2 or "A_log" in n_ or "dt_bias" in n_ or n_.endswith(".D") else decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
step = 0
if os.path.exists(args.ckpt):
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    for b in model.blocks: b.mixer._steps = step
    print(f"resumed from {args.ckpt} at step {step}", flush=True)

train_it, val_it = batches("train", args.seq_len), batches("val", args.seq_len)
model.train(); t0 = time.time(); tok = 0; losses = []
while step < args.steps:
    xb = next(train_it).to(dev, non_blocking=True)
    for g in opt.param_groups: g["lr"] = lr_at(step)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(xb[:, :-1])
    loss = F.cross_entropy(logits.float().view(-1, args.vocab), xb[:, 1:].reshape(-1))
    aux = model.aux()
    (loss + aux).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    step += 1; tok += xb.numel(); losses.append(loss.item())
    if step % args.log_every == 0:
        dt = time.time() - t0
        print(f"step {step} loss {np.mean(losses[-args.log_every:]):.4f} aux {float(aux):.4f} lr {lr_at(step):.2e} {tok/dt/1e3:.1f} ktok/s elapsed {dt/60:.1f}m", flush=True)
    if step % args.val_every == 0 or step == args.steps:
        model.eval(); vl = []
        with torch.no_grad():
            for _ in range(4):
                vb = next(val_it).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(vb[:, :-1])
                vl.append(F.cross_entropy(lg.float().view(-1, args.vocab), vb[:, 1:].reshape(-1)).item())
        print(f"VAL step {step} loss {np.mean(vl):.4f} ppl {math.exp(np.mean(vl)):.1f}", flush=True); model.train()
    if step % args.ckpt_every == 0 or step == args.steps:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, args.ckpt)
print("TRAINING DONE", flush=True)

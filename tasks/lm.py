"""Small language-model trainer for the per-position-loss study.

Model: token embedding (tied head) -> n_layers x [RMSNorm -> mixer] -> RMSNorm -> logits.
Mixer: zoo_smat_mixer.SmatMamba2MR for every arm, so the arms differ only in G:
    --arm mamba2       d=1, lambda off              (plain Mamba-2)
    --arm smat         content-hashed SMAT G on top (paper mask, reset at T/2, v10 recipe by default)
Data: uint16 GPT-2 token memmaps (tasks/data/prep_pg19.py); random windows of --seq_len+1 tokens.
Checkpoints every --ckpt_every steps to --ckpt (resumable: the job limit is 2 h)."""
import argparse, math, os, sys, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "smat_lm")]   # smat, and the flat kernel modules
from smat.mixers import zoology as z
zoo_smat_mixer = z                                   # the pre-reorg name, still used

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="mamba2", choices=["mamba2", "smat", "attn", "gdn"]); ap.add_argument("--mlp_mult", type=int, default=4)   # attn arm: transformer block = attention + MLP(mlp_mult x d_model)
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--d_model", type=int, default=512); ap.add_argument("--n_layers", type=int, default=8)
ap.add_argument("--d_state", type=int, default=64); ap.add_argument("--headdim", type=int, default=64)
ap.add_argument("--seq_len", type=int, default=16384); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--data", default=os.path.join(os.environ.get("SMAT_WORK", "data"), "pg19")); ap.add_argument("--vocab", type=int, default=50257)
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
ap.add_argument("--ldc", default="", help="long-doc corpus dir (lm/prep_ldc.py): fixed order.npy over (source, chunk) rows")
ap.add_argument("--sparse_ops", type=int, default=1); ap.add_argument("--g_bf16", type=int, default=1); ap.add_argument("--hash_ckpt", type=int, default=0)
ap.add_argument("--n_heads_attn", type=int, default=8)
args = ap.parse_args()
if args.bytes: args.vocab = 256
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda"


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w


class Attn(nn.Module):
    """Full causal softmax attention with RoPE (flash via SDPA); the attention baseline."""
    def __init__(self):
        super().__init__()
        self.h = args.n_heads_attn; self.dh = args.d_model // self.h
        self.qkv = nn.Linear(args.d_model, 3 * args.d_model, bias=False); self.out = nn.Linear(args.d_model, args.d_model, bias=False)
        inv = 1.0 / (10000 ** (torch.arange(0, self.dh, 2).float() / self.dh)); self.register_buffer("inv", inv, persistent=False)
    def rope(self, x, T):
        t = torch.arange(T, device=x.device).float(); f = torch.outer(t, self.inv); cos, sin = f.cos()[None, None], f.sin()[None, None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).to(x.dtype)
    def forward(self, x):
        b, T, _ = x.shape
        q, k, v = self.qkv(x).view(b, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, T), self.rope(k, T)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, T, -1))
    def get_auxiliary_loss(self): return 0.0


class GDN(nn.Module):
    """Gated DeltaNet baseline: flash-linear-attention's GatedDeltaNet layer as is (chunk mode, gate, short conv,
    expand_v 2, head_dim 64), i.e. the standard GDN block."""
    def __init__(self, i):
        super().__init__()
        from fla.layers.gated_deltanet import GatedDeltaNet
        self.gdn = GatedDeltaNet(hidden_size=args.d_model, expand_v=2, head_dim=64, num_heads=args.d_model // 64,
                                 mode="chunk", use_gate=True, use_short_conv=True, layer_idx=i)
    def forward(self, x):
        return self.gdn(x)[0]
    def get_auxiliary_loss(self): return 0.0


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        h = args.mlp_mult * args.d_model
        self.up = nn.Linear(args.d_model, h, bias=False); self.down = nn.Linear(h, args.d_model, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.norm = RMSNorm(args.d_model); self.mlp = None
        if args.arm == "attn":                                          # transformer block: attention + MLP (parameter-matched via --mlp_mult)
            self.mixer = Attn(); self.norm2 = RMSNorm(args.d_model); self.mlp = MLP(); return
        if args.arm == "gdn":
            self.mixer = GDN(i); return
        d = 1 if args.arm == "mamba2" else args.d
        codim = args.hash_codim if args.hash_codim is not None else 1         # paper's mask: hyperplane incidence (c=1); c=d-1 is the point read
        self.mixer = z.SmatMamba2MR(args.d_model, layer_idx=i, d=d, d_state=args.d_state, headdim=args.headdim,
                                    lam_act="one", reset=bool(args.reset), g_decay=True, read="chash", hash_mode="point",
                                    hash_codim=codim, hash_conv=bool(args.hash_conv), anneal_steps=args.anneal,
                                    balance_coef=args.balance, balance_gated=bool(args.balance_gated),
                                    hash_lr_scale=args.hash_lr, hash_src=args.hash_src, hash_freeze=bool(args.hash_freeze), hash_conv_width=args.hash_conv_width, hash_ckpt=bool(args.hash_ckpt),
                                    sparse_ops=bool(args.sparse_ops), g_bf16=bool(args.g_bf16))
    def forward(self, x):
        x = x + self.mixer(self.norm(x))
        return x if self.mlp is None else x + self.mlp(self.norm2(x))


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
    if args.ldc:                                                      # fixed order over pre-cut SEQ-token chunks; +1 token from the next chunk's start is
        import json, glob                                             # not available, so the LM target is the chunk shifted by one (n = SEQ - 1 tokens of loss)
        names = sorted(os.path.basename(f)[:-9] for f in glob.glob(f"{args.ldc}/train/*.bin.json"))
        mm = {sp: [np.memmap(f"{args.ldc}/{sp}/{nm}.bin", dtype=np.uint16, mode="r", shape=(json.load(open(f"{args.ldc}/{sp}/{nm}.bin.json"))["n_chunks"], json.load(open(f"{args.ldc}/{sp}/{nm}.bin.json"))["seq"])) for nm in names] for sp in ("train", "val")}
        if split == "train":
            order = np.load(f"{args.ldc}/order.npy"); pos = 0
            while True:
                rows = order[pos:pos + args.batch]; pos = (pos + args.batch) % len(order)
                yield torch.stack([torch.from_numpy(mm["train"][si][ci].astype(np.int64)) for si, ci in rows])
        else:
            rng = np.random.default_rng(1)
            while True:
                yield torch.stack([torch.from_numpy(mm["val"][si][rng.integers(0, len(mm["val"][si]))].astype(np.int64)) for si in rng.integers(0, len(names), args.batch)])
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
# The content-hash modules (per sequence length) are created lazily inside the mixer at the first forward.  Build them
# NOW, before the optimizer sees model.parameters(), or the hash never trains (this bug affected every LM / selective-
# copying SMAT run before 2026-09-12: a frozen random contextual hash).  The dummy forward also lets a checkpoint load.
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    model(torch.zeros(1, args.seq_len, dtype=torch.long, device=dev))
model.train()
n_params = sum(p.numel() for p in model.parameters()); n_emb = model.emb.weight.numel()
print(f"arm={args.arm} d={args.d} layers={args.n_layers} d_model={args.d_model} params={n_params/1e6:.1f}M (non-emb {(n_params-n_emb)/1e6:.1f}M) T={args.seq_len} batch={args.batch}", flush=True)
decay, no_decay = [], []
for n_, p in model.named_parameters():
    (no_decay if p.ndim < 2 or "A_log" in n_ or "dt_bias" in n_ or n_.endswith(".D") else decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
n_opt = sum(len(g["params"]) for g in opt.param_groups); n_model = sum(1 for _ in model.parameters())
n_hash = sum(1 for n_, _ in model.named_parameters() if ".ca." in n_)
assert n_opt == n_model, (n_opt, n_model)
print(f"optimizer covers {n_opt}/{n_model} parameter tensors ({n_hash} content-hash tensors)", flush=True)
step = 0
if os.path.exists(args.ckpt):
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"])
    names = [n_ for n_, _ in model.named_parameters()]
    if ck.get("param_names") and ck["param_names"] != names:                 # parameter order changed since the save: remap Adam state by name
        old_pos = {n_: i for i, n_ in enumerate(ck["param_names"])}
        order = [p_ for g in opt.param_groups for p_ in g["params"]]; pid = {id(p_): i for i, p_ in enumerate(order)}
        cur_names = [None] * len(order)
        for n_, p_ in model.named_parameters(): cur_names[pid[id(p_)]] = n_
        st = ck["opt"]["state"]; ck["opt"]["state"] = {i: st[old_pos[n_]] for i, n_ in enumerate(cur_names) if old_pos.get(n_) in st}
        print("remapped optimizer state by parameter name", flush=True)
    opt.load_state_dict(ck["opt"]); step = ck["step"]
    for b in model.blocks: b.mixer._steps = step
    print(f"resumed from {args.ckpt} at step {step}", flush=True)

train_it, val_it = batches("train", args.seq_len), batches("val", args.seq_len)
model.train(); t0 = time.time(); tok = 0; losses = []
while step < args.steps:
    xb = next(train_it).to(dev, non_blocking=True)
    for g in opt.param_groups: g["lr"] = lr_at(step)
    inp, tgt = (xb, F.pad(xb[:, 1:], (0, 1), value=-100)) if xb.shape[1] == args.seq_len else (xb[:, :-1], xb[:, 1:])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inp)
    loss = F.cross_entropy(logits.float().view(-1, args.vocab), tgt.reshape(-1), ignore_index=-100)
    aux = model.aux()
    (loss + aux).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    step += 1; tok += xb.numel(); losses.append(loss.item())
    if step % args.log_every == 0:
        dt = time.time() - t0
        print(f"step {step} loss {np.mean(losses[-args.log_every:]):.4f} aux {float(aux):.4f} lr {lr_at(step):.2e} {tok/dt/1e3:.1f} ktok/s peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB elapsed {dt/60:.1f}m", flush=True)
    if step % args.val_every == 0 or step == args.steps:
        model.eval(); vl = []
        with torch.no_grad():
            for _ in range(4):
                vb = next(val_it).to(dev)
                vi, vt = (vb, F.pad(vb[:, 1:], (0, 1), value=-100)) if vb.shape[1] == args.seq_len else (vb[:, :-1], vb[:, 1:])
                with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(vi)
                vl.append(F.cross_entropy(lg.float().view(-1, args.vocab), vt.reshape(-1), ignore_index=-100).item())
        print(f"VAL step {step} loss {np.mean(vl):.4f} ppl {math.exp(np.mean(vl)):.1f}", flush=True); model.train()
    if step % args.ckpt_every == 0 or step == args.steps:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args), "param_names": [n_ for n_, _ in model.named_parameters()]}, args.ckpt)
print("TRAINING DONE", flush=True)

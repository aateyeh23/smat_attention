"""Per-position loss (Lin et al. 2025 style): mean NLL at each token position over --n_tokens of held-out
text in sequences of --seq_len, running average with window --window.  Writes <out>.csv and <out>.png."""
import argparse, os, sys, json, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/u/an author/smat_attention/the GPU cluster")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--data", default="/work/hdd/bekw/an author/pg19"); ap.add_argument("--n_tokens", type=float, default=3.9e7)
ap.add_argument("--seq_len", type=int, default=16384); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--window", type=int, default=501); ap.add_argument("--split", default="test")
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cuda"); targs = ck["args"]
sys.argv = ["train_lm2.py", "--ckpt", "/dev/null"] + [f"--{k}={v}" for k, v in targs.items() if k not in ("ckpt", "synthetic") and v is not None and not isinstance(v, bool)]
import importlib.util
spec = importlib.util.spec_from_file_location("tl", "/u/an author/smat_attention/the GPU cluster/lm/train_lm2.py")
# build the model exactly as trained without running the training loop
src = open("/u/an author/smat_attention/the GPU cluster/lm/train_lm2.py").read().split("model = LM().to(dev)")[0]
ns = {}; exec(compile(src, "train_lm2_defs", "exec"), ns)
model = ns["LM"]().cuda(); model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):      # build the per-length hash modules before loading
    model(torch.zeros(1, a.seq_len, dtype=torch.long, device="cuda"))
sd = ck["model"]; msd = model.state_dict()
for k_, v_ in sd.items():                                              # adopt checkpoint shapes for size-only mismatches (alpha table)
    if k_ in msd and msd[k_].shape != v_.shape and k_.endswith("alpha"):
        mod_name, _, pname = k_.rpartition("."); mod = model.get_submodule(mod_name)
        setattr(mod, pname, torch.nn.Parameter(v_.clone().to(msd[k_].device)))
model.load_state_dict(sd)
z = ns["z"]
T = a.seq_len
if targs.get("ldc"):                                                   # LDC corpus: evaluate on the held-out chunks of every source
    import json, glob
    names = sorted(os.path.basename(f)[:-9] for f in glob.glob(f"{targs['ldc']}/val/*.bin.json"))
    parts = [np.memmap(f"{targs['ldc']}/val/{nm}.bin", dtype=np.uint16, mode="r", shape=(json.load(open(f"{targs['ldc']}/val/{nm}.bin.json"))["n_chunks"], T)) for nm in names]
    data = np.concatenate([np.asarray(pt) for pt in parts], 0)          # (n_chunks, T) held-out chunks
    print("LDC val chunks per source:", {nm: len(pt) for nm, pt in zip(names, parts)}, flush=True)
else:
    data = ns["load_split"](a.split)
ldc = data.ndim == 2
n_seq = int(a.n_tokens // T); n_seq = min(n_seq, len(data) if ldc else (len(data) - 1) // T)
print(f"eval {n_seq} sequences x {T} = {n_seq*T/1e6:.1f}M tokens from split {a.split}", flush=True)
acc = torch.zeros(T, dtype=torch.float64, device="cuda"); cnt = 0
with torch.no_grad():
    for s in range(0, n_seq, a.batch):
        ix = list(range(s, min(s + a.batch, n_seq)))
        if ldc:
            xb = torch.stack([torch.from_numpy(data[i].astype(np.int64)) for i in ix]).cuda()
            inp, tgt = xb, F.pad(xb[:, 1:], (0, 1), value=-100)
        else:
            xb = torch.stack([torch.from_numpy(data[i*T:i*T+T+1].astype(np.int64)) for i in ix]).cuda()
            inp, tgt = xb[:, :-1], xb[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(inp)
        nll = F.cross_entropy(lg.float().transpose(1, 2), tgt, reduction="none", ignore_index=-100)   # (b, T); last position 0 in ldc mode
        acc += nll.double().sum(0); cnt += len(ix)
        if (s // a.batch) % 50 == 0: print(f"  {cnt}/{n_seq} seqs, running mean {float(acc.sum()/cnt/T):.4f}", flush=True)
per_pos = (acc / cnt).cpu().numpy()
w = a.window; kern = np.ones(w) / w
smooth = np.convolve(per_pos, kern, mode="valid"); xs = np.arange(len(smooth)) + w // 2
np.savetxt(a.out + ".csv", np.stack([np.arange(T), per_pos], 1), delimiter=",", header="position,nll", comments="")
summary = {"arm": targs["arm"], "d": targs["d"], "mean_nll": float(per_pos.mean()), "ppl": float(math.exp(per_pos.mean())),
           "bpb": float(per_pos.mean() / math.log(2)), "nll_first1k": float(per_pos[:1024].mean()), "nll_last1k": float(per_pos[-1024:].mean()),
           "nll_at": {str(p): float(smooth[min(p, len(smooth)-1)]) for p in (512, 1024, 2048, 4096, 6144, 8000)}}
json.dump(summary, open(a.out + ".json", "w"), indent=1); print(json.dumps(summary), flush=True)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4)); plt.plot(xs, smooth); plt.xlabel("token position"); plt.ylabel(f"loss (running mean, window {w})")
    plt.title(f"{targs['arm']} d={targs['d']}  mean {per_pos.mean():.3f}"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(a.out + ".png", dpi=130)
except Exception as e: print("plot skipped:", e)
print("EVAL DONE", flush=True)

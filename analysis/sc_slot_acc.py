import sys, torch, numpy as np
sys.path.insert(0, "/u/an author/smat_attention/the GPU cluster")
ck_path = sys.argv[1]
ck = torch.load(ck_path, map_location="cuda", weights_only=False); targs = ck["args"]
sys.argv = ["sc_train.py", "--ckpt", "/dev/null"] + [f"--{k}={v}" for k, v in targs.items() if k != "ckpt" and v is not None and not isinstance(v, bool)]
src = open("/u/an author/smat_attention/the GPU cluster/lm/sc_train.py").read().split("model = LM().to(dev)")[0]
ns = {}; exec(compile(src, "sc_defs", "exec"), ns)
model = ns["LM"]().cuda()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): model(torch.zeros(1, ns["args"].seq_len, dtype=torch.long, device="cuda"))
model.load_state_dict(ck["model"]); model.eval()
g = torch.Generator().manual_seed(1234); xe, ye = ns["make_batch"](256, g); xe, ye = xe.cuda(), ye.cuda()
k = ns["args"].n_copy; L = ns["args"].seq_len
hits = torch.zeros(k); first_half = torch.zeros(k); cnt_fh = torch.zeros(k)
with torch.no_grad():
    for i in range(0, 256, 64):
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(xe[i:i+64])
        pred = lg.argmax(-1)[:, L-k:]; tgt = ye[i:i+64][:, L-k:]
        hits += (pred == tgt).float().sum(0).cpu()
        # position of the j-th content token in each sequence: was it in the first half?
        x = xe[i:i+64]; content_pos = torch.stack([torch.nonzero(x[b, :L-k] < ns["args"].n_vocab).flatten() for b in range(x.shape[0])])  # (64, k)
        fh = (content_pos < L // 2).float().cpu(); first_half += ((pred == tgt).float().cpu() * fh).sum(0); cnt_fh += fh.sum(0)
acc = hits / 256
print(f"{ck_path.split('/')[-1]} step {ck['step']}: mean {acc.mean():.3f}")
print("  per slot (1..16): " + " ".join(f"{a:.2f}" for a in acc.tolist()))
print("  acc on tokens that sat in the FIRST half of the prefix: " + " ".join(f"{(h/max(c,1)):.2f}" for h, c in zip(first_half.tolist(), cnt_fh.tolist())))

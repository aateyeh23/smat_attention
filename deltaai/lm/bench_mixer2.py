"""One-layer throughput at LM scale: Mamba-2 vs SMAT chash point read, dense vs segmented ops, fp32 vs bf16."""
import time, torch, sys
sys.path.insert(0, "/u/archerdw/smat_attention/deltaai")
import zoo_smat_mixer as z
dm, T, B = 512, 16384, 4
def bench(mod, name, steps=4):
    mod = mod.cuda().train(); x = torch.randn(B, T, dm, device="cuda", requires_grad=True); torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(steps + 1):
            if i == 1: torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16): y = mod(x)
            y.float().mean().backward()
        torch.cuda.synchronize(); dt = (time.time() - t) / steps
        print(f"{name:40s} {dt*1000:8.1f} ms/step  {B*T/dt/1e6:6.2f} Mtok/s  peak {torch.cuda.max_memory_allocated()/1e9:5.1f} GB", flush=True)
    except torch.OutOfMemoryError:
        print(f"{name:40s} OOM", flush=True)
    del mod, x; torch.cuda.empty_cache()
common = dict(d_state=64, headdim=64, lam_act="one", reset=True, g_decay=True, read="chash", hash_mode="point",
              hash_conv=True, hash_conv_width=16, anneal_steps=1000, balance_coef=0.01, balance_gated=True)
bench(z.SmatMamba2MR(dm, layer_idx=0, d=1, d_state=64, headdim=64, lam_act="one"), "mamba2 (d=1)")
for d in (2, 3, 4):
    c = d - 1
    pass
    bench(z.SmatMamba2MR(dm, layer_idx=0, d=d, hash_codim=c, sparse_ops=True, **common), f"smat d={d} sorted-bmm fp32")
    bench(z.SmatMamba2MR(dm, layer_idx=0, d=d, hash_codim=c, sparse_ops=True, g_bf16=True, **common), f"smat d={d} sorted-bmm bf16")

"""Throughput of one layer at LM scale: Mamba-2 (zoology class) vs SmatMamba2MR chash d=3, d_model 512, T 8192."""
import time, torch, sys
sys.path.insert(0, "/u/archerdw/smat_attention/deltaai")
import zoo_smat_mixer as z
from zoology.mixers.mamba2 import Mamba2
dm, T, B = 512, 16384, 2
def bench(mod, name, steps=5):
    mod = mod.cuda().train(); x = torch.randn(B, T, dm, device="cuda", requires_grad=True)
    for i in range(steps + 2):
        if i == 2: torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16): y = mod(x)
        y.float().mean().backward()
    torch.cuda.synchronize(); dt = (time.time() - t) / steps
    print(f"{name:28s} {dt*1000:7.1f} ms/step  {B*T/dt/1e6:6.2f} Mtok/s  mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB  params {sum(p.numel() for p in mod.parameters())/1e6:.2f}M", flush=True)
    torch.cuda.reset_peak_memory_stats()
bench(Mamba2(dm, d_state=64, headdim=64, layer_idx=0), "mamba2 (zoology, hd64 ds64)")
bench(z.SmatMamba2MR(dm, layer_idx=0, d=1, d_state=64, headdim=64, lam_act="one"), "MR d=1 (== mamba2)")
bench(z.SmatMamba2MR(dm, layer_idx=0, d=3, d_state=64, headdim=64, lam_act="one", reset=True, g_decay=True, read="chash",
                     hash_mode="point", hash_codim=2, hash_conv=True, anneal_steps=1000, balance_coef=0.01, balance_gated=True), "MR chash d=3 (reset, conv)")
bench(z.SmatMamba2MR(dm, layer_idx=0, d=2, d_state=64, headdim=64, lam_act="one", reset=True, g_decay=True, read="chash",
                     hash_mode="point", hash_codim=1, hash_conv=True, anneal_steps=1000, balance_coef=0.01, balance_gated=True), "MR chash d=2 (reset, conv)")

import torch, sys; sys.path.insert(0, "/u/archerdw/smat_attention/deltaai")
import zoo_smat_mixer as z
from torch.profiler import profile, ProfilerActivity
import os
dm, T, B = int(os.environ.get("DM", 512)), int(os.environ.get("T", 16384)), int(os.environ.get("B", 4))
mod = z.SmatMamba2MR(dm, layer_idx=0, d=int(os.environ.get("D", 3)), hash_codim=int(os.environ.get("CODIM", 2)), d_state=64, headdim=64, lam_act="one", reset=True, g_decay=True, read="chash",
                     hash_mode="point", hash_conv=True, hash_conv_width=16, anneal_steps=int(os.environ.get("ANNEAL", 0)), balance_coef=0.01, balance_gated=True,
                     sparse_ops=True, g_bf16=True).cuda().train()
x = torch.randn(B, T, dm, device="cuda", requires_grad=True)
for _ in range(2):
    with torch.autocast("cuda", dtype=torch.bfloat16): y = mod(x)
    y.float().mean().backward()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16): y = mod(x)
        y.float().mean().backward()
    torch.cuda.synchronize()
import smat_pool_ops, time
print("LAST group:", smat_pool_ops.LAST, "LCAP", smat_pool_ops.LCAP)
torch.cuda.synchronize(); t=time.time()
for _ in range(5):
    with torch.autocast("cuda", dtype=torch.bfloat16): y = mod(x)
    y.float().mean().backward()
torch.cuda.synchronize(); print(f"layer step {(time.time()-t)/5*1000:.1f} ms  peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=int(os.environ.get("ROWS", 28))))

import torch, sys; sys.path.insert(0, "/u/archerdw/smat_attention/deltaai")
import zoo_smat_mixer as z
from torch.profiler import profile, ProfilerActivity
dm, T, B = 512, 16384, 4
mod = z.SmatMamba2MR(dm, layer_idx=0, d=3, hash_codim=2, d_state=64, headdim=64, lam_act="one", reset=True, g_decay=True, read="chash",
                     hash_mode="point", hash_conv=True, hash_conv_width=16, anneal_steps=1000, balance_coef=0.01, balance_gated=True,
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
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=28))

import torch, zoo_smat_configs as c
from zoology.model import LanguageModel
m = LanguageModel(c.configs[0].model).cuda(); mx = m.backbone.layers[0].mixer
print(c.configs[0].model.name, "reset", mx.reset, "read", mx.read, "pool", mx.pool, "decay", mx.g_decay, "lam", mx.lam_act, "heads", mx.h)
for T in (64, 128, 256):
    x = torch.randint(0, 8192, (4, T)).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16): y = m(x)
    y.float().mean().backward()
    print(f"T={T} finite {torch.isfinite(y).all().item()} gscale.grad {mx.gscale.grad is not None and torch.isfinite(mx.gscale.grad).all().item()}")
with torch.no_grad():
    u = torch.randn(2, 256, c.configs[0].model.d_model).cuda()
    full = mx(u); half = mx.mixer(u[:, :128])
    print("reset check: first half equals Mamba-2 on the first half alone:", (full[:, :128] - half).abs().max().item())
    mx.gscale.zero_(); sec = mx(u)[:, 128:]; sec_ref = mx.mixer(u[:, 128:])
    print("reset check: G off -> second half equals Mamba-2 restarted at the boundary:", (sec - sec_ref).abs().max().item())
print("SMOKE OK")

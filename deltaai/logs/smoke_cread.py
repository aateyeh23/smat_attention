import torch, zoo_smat_configs as c
from zoology.model import LanguageModel
m = LanguageModel(c.configs[0].model).cuda(); mx = m.backbone.layers[0].mixer
print(c.configs[0].model.name, "pool", mx.pool, "heads", mx.h, "d_state", mx.N)
for T in (64, 128, 256):
    x = torch.randint(0, 8192, (4, T)).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16): y = m(x)
    y.float().mean().backward()
    print(f"T={T} finite {torch.isfinite(y).all().item()} tau.grad {mx.tau.grad is not None and torch.isfinite(mx.tau.grad).all().item()}")
with torch.no_grad():
    mx.alpha.fill_(-60.0); u = torch.randn(2, 256, c.configs[0].model.d_model).cuda()
    print("lambda->0 equals Mamba-2:", (mx(u) - mx.mixer(u)).abs().max().item())
print("SMOKE OK")

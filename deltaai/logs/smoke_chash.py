import os, torch, zoo_smat_configs as c
from zoology.model import LanguageModel
m = LanguageModel(c.configs[0].model).cuda(); mx = m.backbone.layers[0].mixer
print(c.configs[0].model.name, "read", mx.read, "mode", mx.hash_mode, "conv", mx.hash_conv, "shift", mx.hash_shift, "anneal", mx.anneal_steps)
m.train()
for T in (64, 128, 256):
    x = torch.randint(0, 8192, (4, T)).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16): y = m(x)
    aux = sum(mod.get_auxiliary_loss() for mod in m.modules() if hasattr(mod, "get_auxiliary_loss"))
    (y.float().mean() + aux).backward()
    ca = mx.ca[str(T)]
    print(f"T={T} finite {torch.isfinite(y).all().item()} aux {float(aux):.4f} N0 {ca.N0} q {ca.q} mode {ca.mode} hashW.grad {ca.W.grad is not None and torch.isfinite(ca.W.grad).all().item()} gscale.grad {mx.gscale.grad is not None}")
print("params total", sum(p.numel() for p in m.parameters()), "mixer", sum(p.numel() for p in mx.parameters()), "hash", sum(p.numel() for p in mx.ca.parameters()))
print("SMOKE OK")

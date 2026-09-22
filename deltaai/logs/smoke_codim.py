import torch, zoo_smat_configs as c
from zoology.model import LanguageModel
m = LanguageModel(c.configs[0].model).cuda(); mx = m.backbone.layers[0].mixer; m.train()
print(c.configs[0].model.name)
for T in (64, 128, 256):
    x = torch.randint(0, 8192, (4, T)).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16): y = m(x)
    aux = sum(mod.get_auxiliary_loss() for mod in m.modules() if hasattr(mod, "get_auxiliary_loss"))
    (y.float().mean() + aux).backward()
    ca = mx.ca[str(T)]; sp = mx.specs[T]
    print(f"T={T} d_eff={sp.d} q={ca.q} dim={ca.dim} N0={ca.N0} mode={ca.mode} codim={ca.codim} D={getattr(ca,'D',1)} cosets={getattr(ca,'n_cosets',1)} finite={torch.isfinite(y).all().item()} hashW.grad={ca.W.grad is not None}")
print("SMOKE OK")

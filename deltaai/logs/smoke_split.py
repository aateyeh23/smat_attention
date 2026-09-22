import torch, zoo_smat_configs as c, zoo_smat_mixer as z
from zoology.model import LanguageModel
m = LanguageModel(c.configs[0].model).cuda(); mx = m.backbone.layers[0].mixer; m.train()
print(c.configs[0].model.name, "codim", mx.hash_codim, "src", mx.hash_src, "freeze", mx.hash_freeze, "conv", mx.hash_conv)
for T in (64, 128, 256):
    x = torch.randint(0, 8192, (4, T)).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16): y = m(x)
    aux = sum(mod.get_auxiliary_loss() for mod in m.modules() if hasattr(mod, "get_auxiliary_loss"))
    (y.float().mean() + (aux if torch.is_tensor(aux) else 0)).backward()
    mods = mx.ca[str(T)]
    print(f"T={T} finite={torch.isfinite(y).all().item()} groups={[(k, v.heads, v.mode, getattr(v,'n_cosets',1)) for k,v in mods.items()]} emb_seen={'x' in z._EMB}")
print("SMOKE OK")

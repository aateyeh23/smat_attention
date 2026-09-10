import torch, zoo_smat_configs as c
from zoology.model import LanguageModel
import zoology.mixers.mamba2 as zm
print("block patched:", zm.Mamba2Block.__name__)
cfg = c.configs[0]
m = LanguageModel(cfg.model).cuda()
L0 = m.backbone.layers[0]
print(type(L0).__name__, type(L0.mixer).__name__, type(L0.norm).__name__, "params", sum(p.numel() for p in m.parameters()))
x = torch.randint(0, 8192, (4, 256)).cuda()
y = m(x); y.float().mean().backward()
print("fwd/bwd ok", tuple(y.shape), "alpha grad:", getattr(L0.mixer, "alpha", None) is not None and L0.mixer.alpha.grad is not None)
print("in_proj std %.3f  out_proj std %.3f (kaiming ~0.14 / rescaled, not 0.02)" % (L0.mixer.mixer.in_proj.weight.std(), L0.mixer.mixer.out_proj.weight.std()))

import os, torch, zoo_smat_configs as c
from zoology.model import LanguageModel
cfg = c.configs[0]; print("model:", cfg.model.name, "lr", cfg.learning_rate)
m = LanguageModel(cfg.model).cuda()
mx = m.backbone.layers[0].mixer
print(type(mx).__name__, "heads", mx.h, "headdim", mx.p, "offsets", mx.offsets)
print("params total", sum(p.numel() for p in m.parameters()), " mixer", sum(p.numel() for p in mx.parameters()),
      " G-only", sum(p.numel() for n_, p in mx.named_parameters() if n_.startswith(("lam_w", "alpha"))))
for T in (64, 128, 256):
    x = torch.randint(0, 8192, (4, T)).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x)
    y.float().mean().backward()
    spec = mx.specs.get(T); print(f"T={T} out {tuple(y.shape)} finite {torch.isfinite(y).all().item()} B={getattr(spec,'B',None)} alpha.grad {mx.alpha.grad is not None and torch.isfinite(mx.alpha.grad).all().item()}")
# equivalence check: with lambda forced to 0 the layer must equal plain Mamba-2 (same weights)
with torch.no_grad():
    mx.alpha.fill_(-60.0)
    x = torch.randint(0, 8192, (2, 256)).cuda()
    u = torch.randn(2, 256, cfg.model.d_model).cuda()
    y_mr = mx(u); y_m2 = mx.mixer(u)
    print("lambda->0 equals Mamba-2:", (y_mr - y_m2).abs().max().item())
print("SMOKE OK")

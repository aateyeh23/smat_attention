import os, sys, torch
import zoo_smat_configs as c
from zoology.model import LanguageModel
m = LanguageModel(c.configs[0].model).cuda()
for T in (64, 256):
    x = torch.randint(0, 8192, (4, T)).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16): y = m(x)
    y.float().mean().backward(); torch.cuda.synchronize()
print("OK", c.configs[0].model.name, flush=True)

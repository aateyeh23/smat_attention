import torch, zoo_smat_configs as c
from zoology.model import LanguageModel
torch.autograd.set_detect_anomaly(True)
m = LanguageModel(c.configs[0].model).cuda(); print(c.configs[0].model.name)
x = torch.randint(0, 8192, (4, 64)).cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = m(x)
y.float().mean().backward(); print("DECAY SMOKE OK")

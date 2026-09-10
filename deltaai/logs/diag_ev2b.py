import sys, torch
torch.manual_seed(0)
mode = sys.argv[1]
from fla.layers.gated_deltanet import GatedDeltaNet
import zoo_smat_mixer as z
u = torch.randn(4, 64, 16, device="cuda", requires_grad=True)
if mode == "layer_bf16":
    g = GatedDeltaNet(hidden_size=16, expand_v=2, head_dim=16, num_heads=1, layer_idx=0).cuda().train()
    with torch.autocast("cuda", dtype=torch.bfloat16): o = g(u)[0]
    o.float().mean().backward(); torch.cuda.synchronize(); print("fla layer h1 ev2 bf16 OK")
elif mode == "wrap_fp32":
    m = z.SmatGDN(16, d=1, headdim=16, expand_v=2, n_heads=1).cuda().train()
    o = m(u); o.float().mean().backward(); torch.cuda.synchronize(); print("wrapper d1 h1 ev2 fp32 OK")
elif mode == "wrap_bf16":
    m = z.SmatGDN(16, d=1, headdim=16, expand_v=2, n_heads=1).cuda().train()
    with torch.autocast("cuda", dtype=torch.bfloat16): o = m(u)
    o.float().mean().backward(); torch.cuda.synchronize(); print("wrapper d1 h1 ev2 bf16 OK")
elif mode == "wrap_fwd_only":
    m = z.SmatGDN(16, d=1, headdim=16, expand_v=2, n_heads=1).cuda().train()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): o = m(u)
    torch.cuda.synchronize(); print("wrapper fwd-only bf16 OK")

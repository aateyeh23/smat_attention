import sys, torch
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
mode = sys.argv[1]
torch.manual_seed(0)
if mode == "layer":       # fla's own layer, its own forward, 1 head, head_dim 16, expand_v 2
    g = GatedDeltaNet(hidden_size=16, expand_v=2, head_dim=16, num_heads=1, layer_idx=0).cuda().train()
    u = torch.randn(4, 64, 16, device="cuda", requires_grad=True)
    o = g(u)[0]; o.float().mean().backward(); torch.cuda.synchronize(); print("layer h1 ev2 OK")
elif mode == "layer_h2":
    g = GatedDeltaNet(hidden_size=16, expand_v=2, head_dim=16, num_heads=2, layer_idx=0).cuda().train()
    u = torch.randn(4, 64, 16, device="cuda", requires_grad=True)
    o = g(u)[0]; o.float().mean().backward(); torch.cuda.synchronize(); print("layer h2 ev2 OK")
elif mode == "op_fwd":    # raw op, forward only, H=1 K=16 V=32
    b, l, h, dk, dv = 4, 64, 1, 16, 32
    q, k, v = (torch.randn(b, l, h, d, device="cuda", requires_grad=True) for d in (dk, dk, dv))
    gg = torch.randn(b, l, h, device="cuda"); beta = torch.randn(b, l, h, device="cuda")
    o, _ = chunk_gated_delta_rule(q, k, v, gg, beta, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, state_v_first=True)
    torch.cuda.synchronize(); print("op fwd H1 K16 V32 OK"); o.float().mean().backward(); torch.cuda.synchronize(); print("op bwd H1 K16 V32 OK")
elif mode == "op_v16":
    b, l, h, dk, dv = 4, 64, 1, 16, 16
    q, k, v = (torch.randn(b, l, h, d, device="cuda", requires_grad=True) for d in (dk, dk, dv))
    gg = torch.randn(b, l, h, device="cuda"); beta = torch.randn(b, l, h, device="cuda")
    o, _ = chunk_gated_delta_rule(q, k, v, gg, beta, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, state_v_first=True)
    o.float().mean().backward(); torch.cuda.synchronize(); print("op H1 K16 V16 OK")
elif mode == "op_h2v32":
    b, l, h, dk, dv = 4, 64, 2, 16, 32
    q, k, v = (torch.randn(b, l, h, d, device="cuda", requires_grad=True) for d in (dk, dk, dv))
    gg = torch.randn(b, l, h, device="cuda"); beta = torch.randn(b, l, h, device="cuda")
    o, _ = chunk_gated_delta_rule(q, k, v, gg, beta, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, state_v_first=True)
    o.float().mean().backward(); torch.cuda.synchronize(); print("op H2 K16 V32 OK")

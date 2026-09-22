"""Check the actual width-16 MQAR model before submitting long runs."""
import torch
from zoo_mamba_reset_configs import MqarMambaReset

torch.manual_seed(123)
u = torch.randn(2, 256, 16, device="cuda")
base = MqarMambaReset(16, d=1).cuda().eval()
assert base.mixer.headdim == 16 and base.mixer.d_state == 16 and base.mixer.nheads == 2
torch.testing.assert_close(base(u), base.mixer(u), rtol=2e-4, atol=2e-4)
print("PASS d=1 matches native Mamba-2", flush=True)
for d in (2, 3, 4):
    m = MqarMambaReset(16, d=d).cuda().eval()
    assert m.mixer.headdim == 16 and m.mixer.d_state == 16 and m.mixer.nheads == 2
    count = len(list(m.parameters()))
    with torch.no_grad():
        m.g_write_proj.bias.fill_(-100)
        closed = m(u)
        changed = u.clone(); changed[:, :128] = torch.randn_like(changed[:, :128])
        torch.testing.assert_close(closed[:, 128:], m(changed)[:, 128:], rtol=0, atol=0)
        m.g_write_proj.bias.fill_(-2.197224577)
    m.train()
    out = m(u)
    assert (out[:, 128:] - closed[:, 128:]).abs().max() > 1e-6
    (out.square().mean() + m.get_auxiliary_loss()).backward()
    grad = m.g_write_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0
    hash_grads = [p.grad for p in m.ca.parameters() if p.grad is not None]
    assert hash_grads and max(g.abs().max().item() for g in hash_grads) > 0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    for length in (64, 128, 256):
        with torch.no_grad():
            assert torch.isfinite(m(u[:, :length])).all()
        assert m._spec(length, u.device).dim + 1 == d
    assert count == len(list(m.parameters()))
    print(f"PASS d={d}: reset isolation, G contribution, gate/hash gradients, all lengths registered", flush=True)
print("ALL MQAR MAMBA RESET TESTS PASSED", flush=True)

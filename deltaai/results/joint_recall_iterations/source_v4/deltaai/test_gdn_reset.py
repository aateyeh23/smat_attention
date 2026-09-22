"""Check GDN equivalence, genuine state/conv reset, and sparse-memory gradients."""
import os
import torch
from zoo_smat_gdn import SmatGDNReset

torch.manual_seed(123)
width = int(os.environ.get("ZOO_DM", "32"))
u = torch.randn(2, 256, width, device="cuda")
base = SmatGDNReset(width, d=1).cuda().train()
torch.testing.assert_close(base(u), base.gdn(u)[0], rtol=2e-4, atol=2e-4)
print("PASS d=1 matches standard GDN", flush=True)
for d in (2, 3, 4):
    m = SmatGDNReset(width, d=d).cuda().train()
    n_params = len(list(m.parameters()))
    m.disable_g = True
    a = m(u)
    changed = u.clone(); changed[:, :128] = torch.randn_like(changed[:, :128])
    torch.testing.assert_close(a[:, 128:], m(changed)[:, 128:], rtol=0, atol=0)
    folded = m.gdn(u.reshape(4, 128, width))[0].reshape(2, 256, width)
    torch.testing.assert_close(a, folded, rtol=2e-4, atol=2e-4)
    m.disable_g = False
    y = m(u)
    assert (y[:, 128:] - a[:, 128:]).abs().max() > 1e-6
    torch.testing.assert_close(y[:, :128], a[:, :128], rtol=0, atol=0)
    loss = (y * torch.randn_like(y)).sum() + m.get_auxiliary_loss()
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
    grads = [p.grad for name, p in m.named_parameters() if ".ca." in "." + name and p.grad is not None]
    assert grads and max(p.abs().max().item() for p in grads) > 0
    torch.optim.AdamW(m.parameters(), lr=1e-3).step()
    for length in (64, 128, 256):
        with torch.no_grad():
            result = m(u[:, :length])
        assert torch.isfinite(result).all()
        spec = m._spec(length, u.device)
        assert spec is not None and spec.dim + 1 == d, (d, length)
    assert n_params == len(list(m.parameters())), "hash parameters were created after optimizer setup"
    print(f"PASS d={d}: true reset, nonzero G, finite gradients, trained hash registered for all mixture lengths", flush=True)
print("ALL GDN RESET TESTS PASSED", flush=True)

"""Residual G writes must exactly recover shared writes at initialization."""
import argparse
import torch
from smat.mixers.zoology import SmatMamba2MR


def make(mode, device):
    torch.manual_seed(91)
    model = SmatMamba2MR(64, d=2, d_state=16, headdim=64, reset=True,
        lam_act="sigmoid", hash_src="ssm", read="chash", hash_codim=1,
        hash_conv=True, g_decay=True, hash_ckpt=True, sparse_ops=True,
        g_bf16=True, g_write_mode=mode, g_write_init=0.1)
    model._spec(128, torch.device("cpu"))
    model.specs.clear()
    return model.to(device)


def check(cpu_only=False):
    device = "cpu" if cpu_only else "cuda"
    shared = make("shared", device)
    shared_rng = torch.get_rng_state()
    corrected = make("residual", device)
    assert torch.equal(shared_rng, torch.get_rng_state())
    for name, param in shared.named_parameters():
        torch.testing.assert_close(param, dict(corrected.named_parameters())[name], rtol=0, atol=0)
    u = torch.randn(2, 128, 64, device=device)
    dt = torch.rand(2, 128, corrected.h, device=device, requires_grad=True)
    w = corrected._g_write_weights(u, dt)
    torch.testing.assert_close(w, dt, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(w.sum(), dt)[0], torch.ones_like(dt), rtol=0, atol=0)
    print("PASS matching common parameters/RNG; exact shared write values and dt gradients", flush=True)
    if cpu_only:
        return
    shared.eval(); corrected.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(shared(u), corrected(u), rtol=0, atol=0)
    corrected.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = corrected(u)
    (out[:, 64:].float().square().mean() + corrected.get_auxiliary_loss()).backward()
    for p in corrected.g_write_proj.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0
    assert all(torch.isfinite(p.grad).all() for p in corrected.parameters() if p.grad is not None)
    torch.optim.AdamW(corrected.parameters(), lr=1e-3).step()
    assert not torch.equal(corrected._g_write_weights(u, dt), dt)
    restored = make("residual", device)
    restored.load_state_dict(corrected.state_dict())
    corrected.eval(); restored.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(restored(u), corrected(u), rtol=0, atol=0)
    print("PASS exact shared forward at initialization; finite correction gradients, learning and checkpoint", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    check(parser.parse_args().cpu_only)

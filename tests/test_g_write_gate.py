"""GPU checks for independent G writes, including the production sparse/bf16 path."""
import argparse
import torch
from smat.mixers.zoology import SmatMamba2MR


def make(mode, sparse=False, device="cuda", d=3):
    return SmatMamba2MR(
        64, d=d, d_state=16, headdim=64, reset=True,
        lam_act="one", hash_src="ssm", read="chash", hash_codim=1,
        hash_conv=True, g_decay=True, hash_ckpt=True,
        sparse_ops=sparse, g_bf16=sparse, g_write_mode=mode,
        g_write_init=0.5,
    ).to(device)


def check_cpu(d=3):
    torch.manual_seed(91)
    shared = make("shared", device="cpu", d=d)
    rng = torch.get_rng_state()
    torch.manual_seed(91)
    gated = make("independent", device="cpu", d=d)
    assert torch.equal(rng, torch.get_rng_state())
    for name, param in shared.named_parameters():
        torch.testing.assert_close(param, dict(gated.named_parameters())[name], rtol=0, atol=0)
    u = torch.randn(2, 8, 64)
    dts = torch.rand(2, 8, gated.h, requires_grad=True)
    assert shared._g_write_weights(u, dts) is dts
    writes = gated._g_write_weights(u, dts)
    torch.testing.assert_close(writes, torch.full_like(writes, 0.5))
    assert torch.autograd.grad(writes.sum(), dts, allow_unused=True, retain_graph=True)[0] is None
    # Optimizing selection must update the new parameters without touching dt.
    optimizer = torch.optim.SGD(gated.parameters(), lr=0.1)
    target = (u[..., :gated.h] > 0).float()
    loss = torch.nn.functional.binary_cross_entropy(writes, target)
    loss.backward()
    assert dts.grad is None
    assert gated.g_write_proj.weight.grad.abs().max() > 0
    optimizer.step()
    assert torch.nn.functional.binary_cross_entropy(gated._g_write_weights(u, dts), target) < loss
    restored = make("independent", device="cpu", d=d)
    restored.load_state_dict(gated.state_dict())
    torch.testing.assert_close(restored._g_write_weights(u, dts), gated._g_write_weights(u, dts), rtol=0, atol=0)
    print("PASS CPU: matching init, independent writes, selection learning, checkpoint", flush=True)


def check(sparse, d=3):
    # Adding the gate must not change the initial common parameters or RNG stream.
    torch.manual_seed(91)
    shared = make("shared", sparse, d=d)
    rng = torch.get_rng_state()
    torch.manual_seed(91)
    gated = make("independent", sparse, d=d)
    assert torch.equal(rng, torch.get_rng_state())
    for name, param in shared.named_parameters():
        torch.testing.assert_close(param, dict(gated.named_parameters())[name], rtol=0, atol=0)

    u = torch.randn(2, 128, 64, device="cuda")
    dts = torch.rand(2, 128, gated.h, device="cuda", requires_grad=True)
    assert shared._g_write_weights(u, dts) is dts
    weights = gated._g_write_weights(u, dts)
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5))
    assert torch.autograd.grad(weights.sum(), dts, allow_unused=True)[0] is None

    # Make decay slow enough that cross-boundary contributions are measurable.
    with torch.no_grad():
        gated.mixer.A_log.fill_(-2)
        gated.mixer.dt_bias.fill_(-4)
        gated.mixer.in_proj.weight[-gated.h:].zero_()
    gated.eval()
    raw = []
    hook = gated.mixer.norm.register_forward_pre_hook(lambda module, inputs: raw.append(inputs[0].detach().clone()))
    for bias in (-100., 0., 100.):
        with torch.no_grad():
            gated.g_write_proj.bias.fill_(bias)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gated(u)
    closed, half, opened = raw
    assert (opened[:, 64:] - closed[:, 64:]).abs().max() > 1e-5, "vacuous: no G signal"
    torch.testing.assert_close(opened[:, :64], closed[:, :64], rtol=0, atol=0)
    expected = (closed[:, 64:].float() + opened[:, 64:].float()) / 2
    # Each branch is rounded before bf16 residual addition; cancellation makes
    # a tolerance relative only to the final sum too strict.
    scale = closed[:, 64:].float().abs() + opened[:, 64:].float().abs()
    bound = 2 * torch.finfo(closed.dtype).eps * scale + 1e-6
    assert ((half[:, 64:].float() - expected).abs() <= bound).all(), "G is not linear in its write weights"
    # With G closed and the recurrence reset, the first half cannot affect the second.
    with torch.no_grad():
        gated.g_write_proj.bias.fill_(-100)
        changed = u.clone(); changed[:, :64] = torch.randn_like(changed[:, :64])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gated(changed)
    torch.testing.assert_close(raw[-1][:, 64:], closed[:, 64:], rtol=0, atol=0)
    hook.remove()

    gated.train()
    with torch.no_grad():
        gated.g_write_proj.bias.zero_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = gated(u)
    (out[:, 64:].float() * torch.randn_like(out[:, 64:].float())).sum().backward()
    for param in (gated.g_write_proj.weight, gated.g_write_proj.bias):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().max() > 0
    torch.optim.AdamW(gated.parameters(), lr=1e-3).step()
    # A checkpoint round trip must include the new trainable gate.
    restored = make("independent", sparse, d=d)
    restored.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        restored(u)  # materialize the hash before loading
    restored.load_state_dict(gated.state_dict())
    gated.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(restored(u), gated(u), rtol=0, atol=0)
    print(f"PASS sparse/bf16={sparse}: matching init, independent writes, G scaling, reset isolation, gradients, checkpoint", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--d", type=int, choices=[2, 3, 4], default=3)
    args = parser.parse_args()
    check_cpu(args.d)
    if not args.cpu_only:
        check(False, args.d)
        check(True, args.d)
        print("ALL G WRITE TESTS PASSED", flush=True)

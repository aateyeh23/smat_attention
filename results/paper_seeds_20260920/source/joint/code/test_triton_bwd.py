"""Check the fused backward passes against the torch path, then time them.

Forward and every gradient (Phi, Psi, Vb) are compared between
``smat_attention(backend=torch)`` and the patched ``backend=auto`` at fp32.  The
kernels use TF32 dots, so agreement is at the 1e-3 level, not 1e-6; the
tolerance below is set accordingly and the max relative error is printed so a
regression is visible rather than merely passing.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smat"))
from smat_mask import build_mask                              # noqa: E402
from smat_attn import smat_attention, to_device, make_phi, featurise   # noqa: E402
import smat_triton_bwd                                        # noqa: E402

dev = torch.device("cuda")
torch.manual_seed(0)


def run(spec, Phi, Psi, Vb, backend):
    Phi = Phi.detach().clone().requires_grad_(True)
    Psi = Psi.detach().clone().requires_grad_(True)
    Vb = Vb.detach().clone().requires_grad_(True)
    if backend == "torch":
        o = smat_attention(Phi, Psi, Vb, spec, chunk=128, acc_dtype=torch.float32,
                           scan_backend="torch", incidence_backend="torch")
    else:
        o = smat_attention(Phi, Psi, Vb, spec, chunk=128, acc_dtype=torch.float32)
    # a fixed random projection so the loss is a generic linear functional
    torch.manual_seed(1)
    w = torch.randn_like(o)
    (o * w).sum().backward()
    return o.detach(), Phi.grad, Psi.grad, Vb.grad


def rel(a, b):
    if a is None:
        return float("nan")                     # no gradient at all
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def check(T, d, nb=4, r=64, dqk=64, dv=64, tol=2e-2):
    spec = to_device(build_mask(T, d, chunk=128), dev)
    phi = make_phi(dqk, r, device=dev, dtype=torch.float32, seed=0)
    Q = torch.randn(nb, T, dqk, device=dev) * 0.3
    K = torch.randn(nb, T, dqk, device=dev) * 0.3
    V = torch.randn(nb, T, dv, device=dev)
    Phi, Psi, Vb = featurise(Q, K, V, phi)

    ref = run(spec, Phi, Psi, Vb, "torch")
    smat_triton_bwd.patch()
    try:
        got = run(spec, Phi, Psi, Vb, "auto")
    finally:
        smat_triton_bwd.unpatch()
    errs = [rel(g, f) for g, f in zip(got, ref)]
    ok = all(e < tol for e in errs) and all(torch.isfinite(g).all() for g in got)
    print(f"  T={T:5d} d={d}  rel err  out {errs[0]:.1e}  dPhi {errs[1]:.1e}  "
          f"dPsi {errs[2]:.1e}  dVb {errs[3]:.1e}   {'ok' if ok else 'FAIL'}")
    return ok


def unpatched_has_no_grad(T=2048, d=3):
    """Sanity: without the patch, backend=auto silently drops the kernel gradients."""
    spec = to_device(build_mask(T, d, chunk=128), dev)
    Phi = torch.rand(2, T, 64, device=dev) + 0.5
    Psi = torch.rand(2, T, 64, device=dev) + 0.5
    Vb = torch.randn(2, T, 65, device=dev)
    ref = run(spec, Phi, Psi, Vb, "torch")
    got = run(spec, Phi, Psi, Vb, "auto")
    print(f"  unpatched auto vs torch: rel err out {rel(got[0], ref[0]):.1e}  "
          f"dPsi {rel(got[2], ref[2]):.1e}  dVb {rel(got[3], ref[3]):.1e}  (grads expected to be wrong)")


def bench(T, d, nbh=32, r=64, p=65, iters=20):
    spec = to_device(build_mask(T, d, chunk=128), dev)
    Phi = (torch.rand(nbh, T, r, device=dev) + 0.5).requires_grad_(True)
    Psi = (torch.rand(nbh, T, r, device=dev) + 0.5).requires_grad_(True)
    Vb = torch.randn(nbh, T, p, device=dev).requires_grad_(True)
    res = {}
    for name in ("torch", "triton"):
        if name == "triton":
            smat_triton_bwd.patch()
        kw = (dict(scan_backend="torch", incidence_backend="torch") if name == "torch" else {})
        for _ in range(3):
            o = smat_attention(Phi, Psi, Vb, spec, chunk=128, acc_dtype=torch.float32, **kw)
            o.sum().backward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(iters):
            o = smat_attention(Phi, Psi, Vb, spec, chunk=128, acc_dtype=torch.float32, **kw)
            o.sum().backward()
        torch.cuda.synchronize()
        res[name] = ((time.time() - t0) / iters * 1000, torch.cuda.max_memory_allocated() / 2**20)
        if name == "triton":
            smat_triton_bwd.unpatch()
    (tt, tm), (kt, km) = res["torch"], res["triton"]
    print(f"  T={T:5d} d={d} nb*h={nbh}: fwd+bwd  torch {tt:6.1f} ms {tm:6.0f} MB   "
          f"triton {kt:6.1f} ms {km:6.0f} MB   speedup {tt/kt:.2f}x")


if __name__ == "__main__":
    print("correctness vs torch path (fp32, TF32 kernels)")
    unpatched_has_no_grad()
    ok = True
    for T, d in [(1024, 1), (1024, 2), (2048, 3), (4096, 1), (4096, 2), (4096, 3), (4096, 4), (8192, 3)]:
        ok &= check(T, d)
    print("ALL OK" if ok else "SOME FAILED")
    print("\ntiming at the LM shapes (nb*heads=32, r=64, p=65), fwd+bwd")
    for T, d in [(4096, 1), (4096, 2), (4096, 3), (8192, 3)]:
        bench(T, d)

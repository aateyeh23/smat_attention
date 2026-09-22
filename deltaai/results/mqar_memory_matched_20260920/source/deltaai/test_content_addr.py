"""Exactness check: the content-addressed long-range branch must equal dense
masked kernel attention under the mask its own hash induces."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smat"))
from smat_mask import build_mask
from content_addr import ContentAssign

torch.manual_seed(0)


def check(T, d, mode, nb=3, h=2, dh=8, r=5, p=4, chunk=32, tol=2e-5):
    spec = build_mask(T, d, chunk=chunk)
    n, m = spec.n, spec.T_R
    ca = ContentAssign(h, dh, spec, mode=mode).double()
    B = nb * h
    u = torch.randn(nb, T, dh).double()
    Phi, Psi = torch.rand(B, T, r).double(), torch.rand(B, T, r).double()
    Vb = torch.randn(B, T, p).double()

    got = ca(u, Phi, Psi, Vb, n)                           # (B, m, p)

    # reference: hard cells -> dense G -> masked kernel attention
    kc = ca._cells(u[:, :n], plant_at=True).argmax(-1)         # (B, n)
    qc = ca._cells(u[:, n:]).argmax(-1)                        # (B, m)
    if ca.mode == "point":
        G = (qc.unsqueeze(2) == kc.unsqueeze(1)).double()       # (B, m, n)
    else:
        dl = torch.einsum("blm,hem->bhle", u[:, n:], ca.Wd)
        e = dl.reshape(B, m, ca.D).argmax(-1)                   # (B, m)
        po = torch.einsum("bix,eox->bieo", ca._cells(u[:, n:]), ca.M.double())
        o = po.gather(2, e[:, :, None, None].expand(B, m, 1, ca.q)).squeeze(2).argmax(-1)
        Mi = ca.M.double()[e, o]                                # (B, m, N0)
        G = Mi.gather(2, kc.unsqueeze(1).expand(B, m, n))       # (B, m, n)

    S = torch.einsum("bir,bjr->bij", Phi[:, n:], Psi[:, :n])
    ref = (S * G) @ Vb[:, :n]
    err = (got - ref).abs().max().item()
    dens = G.mean().item()
    print(f"T={T} d={d} mode={mode:5s} q={spec.q} N0={spec.N0} B={spec.B}  "
          f"max|err|={err:.2e}  G density={dens:.3f}", flush=True)
    assert err < tol, f"MISMATCH {err}"
    assert dens > 0, "G is empty -- the test is vacuous"


def check_grad(T, d, mode, chunk=32):
    """the straight-through path must deliver a nonzero gradient to the hash."""
    spec = build_mask(T, d, chunk=chunk)
    ca = ContentAssign(2, 8, spec, mode=mode).double()
    B, n = 6, spec.n
    u = torch.randn(B // 2, T, 8).double()
    Phi, Psi = torch.rand(B, T, 5).double(), torch.rand(B, T, 5).double()
    ca(u, Phi, Psi, torch.randn(B, T, 4).double(), n).sum().backward()
    gW = ca.W.grad.abs().max().item()
    print(f"T={T} d={d} mode={mode:5s} max|dL/dW|={gW:.3e}", flush=True)
    assert gW > 0, "no gradient reaches the profile hash"


if __name__ == "__main__":
    for T, d in ((128, 2), (128, 3), (512, 2), (512, 3)):
        for mode in ("point", "plane"):
            check(T, d, mode, chunk=max(16, T // 4))
    for T, d in ((128, 2), (128, 3)):
        for mode in ("point", "plane"):
            check_grad(T, d, mode, chunk=32)
    print("\nALL OK")

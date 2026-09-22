"""Fixed read budget, dense-reference gradients, and full GDN GPU integration."""
import argparse
import io
import torch
from content_addr import ContentAssign
from smat_mask import build_mask


def check_reference():
    for d in (2, 3, 4):
        torch.manual_seed(123)
        ca = ContentAssign(1, 16, build_mask(64, d, chunk=16), codim=1).double()
        ca.enable_topk_reads(4, 16)
        ca = ca.double()
        ca.sparse_ops, ca.anneal = True, 1.0
        u = torch.randn(2, 64, 16, dtype=torch.double, requires_grad=True)
        q, k, v = [torch.randn(2, 64, 4, dtype=torch.double, requires_grad=True) for _ in range(3)]
        output = ca(u, q, k, v, 32)
        wi, ww = ca._sparse
        assert wi.shape[-1] == 1
        torch.testing.assert_close(ww, torch.ones_like(ww), atol=0, rtol=0)
        writes = ww.new_zeros(2, 32, ca.N0).scatter_add(-1, wi, ww)
        memory = torch.einsum('bnx,bnr,bnp->bxrp', writes, k[:, :32], v[:, :32])
        if ca.mode == 'plane':
            memory = torch.einsum('eox,bxrp->beorp', ca.M, memory).flatten(1, 2)
        ri, rw = ca._topk_reads(u[:, 32:])
        assert ri.shape[-1] == 4 and (ri.sort(-1).values.diff(dim=-1) > 0).all()
        assert (rw > 0).all()
        torch.testing.assert_close(rw.sum(-1), torch.ones_like(rw[..., 0]))
        reads = rw.new_zeros(2, 32, memory.shape[1]).scatter_add(-1, ri, rw)
        reference = torch.einsum('bix,bir,bxrp->bip', reads, q[:, 32:], memory)
        torch.testing.assert_close(output, reference, atol=1e-10, rtol=1e-10)
        inputs = (u, q, k, v, ca.W, ca.W_read, ca.b_read)
        probe = torch.randn_like(output)
        actual_grads = torch.autograd.grad((output * probe).sum(), inputs, retain_graph=True)
        reference_grads = torch.autograd.grad((reference * probe).sum(), inputs)
        for actual, expected in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        # All four distinct types, including different directions when available,
        # must be reachable by the categorical selector.
        with torch.no_grad():
            ca.W_read.zero_()
            ca.b_read.fill_(-10)
            chosen = torch.tensor([0, 1, ca.n_cosets, ca.n_cosets + 1]) if ca.mode == 'plane' else torch.arange(4)
            ca.b_read[:, chosen] = 10
            selected, _ = ca._topk_reads(u[:, 32:])
            assert torch.equal(selected.sort(-1).values, chosen.sort().values.expand_as(selected))
        print(f'PASS CPU d={d}: exact 1 write/4 reads, dense outputs/gradients, distinct types', flush=True)


def check_gpu():
    from zoo_smat_gdn import SmatGDNReset
    assert torch.cuda.is_available()
    for d in (2, 3, 4):
        torch.manual_seed(123)
        kwargs = dict(d= d, headdim=16, n_heads=1, expand_v=1,
                      memory_update='delta', memory_key_mode='tied_causal', memory_read_k=4)
        model = SmatGDNReset(16, **kwargs).cuda()
        assert (model.h, model.r, model.p) == (1, 16, 16)
        parameter_ids = {id(p) for p in model.parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        for length in (64, 128, 256):
            u = torch.randn(2, length, 16, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            y = model(u)
            (y.square().mean() + model.get_auxiliary_loss()).backward()
            ca = next(iter(model.ca[str(length)].values()))
            assert ca.dim == d - 1 and ca.last_write_idx.shape[-1] == 1
            assert ca.last_read_idx.shape[-1] == 4
            assert (ca.last_read_idx.sort(-1).values.diff(dim=-1) > 0).all()
            assert (ca.last_read_weights > 0).all()
            torch.testing.assert_close(ca.last_read_weights.sum(-1), torch.ones_like(ca.last_read_weights[..., 0]))
            assert torch.isfinite(y).all()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            for p in (ca.W, ca.W_read, model.memory_qk.weight, model.kconv.weight):
                assert p.grad is not None and p.grad.abs().max() > 0
            assert {id(p) for p in model.parameters()} == parameter_ids
            optimizer.step()
            with torch.no_grad():
                train_out = model(u)
                model.eval()
                eval_out = model(u)
                torch.testing.assert_close(train_out, eval_out, atol=2e-5, rtol=2e-5)
                model.disable_g = True
                baseline = model(u)
                changed = u.clone(); changed[:, :length // 2] += torch.randn_like(changed[:, :length // 2])
                isolated = model(changed)
                torch.testing.assert_close(baseline[:, length // 2:], isolated[:, length // 2:], atol=0, rtol=0)
                assert (eval_out[:, length // 2:] - baseline[:, length // 2:]).abs().max() > 0
                model.disable_g = False
                model.train()
            print(f'PASS GPU d={d} length={length} q={ca.q} profiles={ca.N0}: 16x16 state, routing, backward, train/eval, reset', flush=True)
        state = io.BytesIO()
        torch.save(model.state_dict(), state); state.seek(0)
        restored = SmatGDNReset(16, **kwargs).cuda()
        restored.load_state_dict(torch.load(state, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(model.eval()(u), restored.eval()(u), atol=0, rtol=0)
        print(f'PASS GPU d={d}: checkpoint roundtrip', flush=True)
    print('ALL FOUR-READ CHECKS PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    check_reference()
    if args.gpu:
        check_gpu()

"""GPU numerical validation and matched forward/backward timing for fused reads."""
import copy
import json
import torch
import triton
from content_addr import _grouped_planes
from smat_read_triton import read_topk, top4_softmax
from smat_aggregation_experiments import incidence
from smat_pool_ops import read_sorted


def compare(label, actual, expected, atol=3e-4, rtol=3e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    print(json.dumps(dict(check=label, max_error=(actual-expected).abs().max().item())), flush=True)


def check_read(b, m, n, r, p, hot=False):
    q = torch.randn(b, m, r, device='cuda', requires_grad=True)
    w = torch.randn(b, m, 4, device='cuda', requires_grad=True)
    f = torch.randn(b, n, r, p, device='cuda', requires_grad=True)
    idx = torch.randint(1 if hot else n, (b, m, 4), device='cuda')
    # Independent float64 GPU reference, including duplicate routes.
    gathered = f.double()[torch.arange(b, device='cuda')[:, None, None], idx]
    expected = torch.einsum('bmr,bmkrp,bmk->bmp', q.double(), gathered, w.double()).float()
    actual = read_topk(q, idx, w, f)
    dy = torch.randn_like(actual)
    compare('read_output', actual, expected)
    ga = torch.autograd.grad(actual, (q, w, f), dy)
    ge = torch.autograd.grad(expected, (q, w, f), dy)
    for label, a, e in zip(('query', 'weights', 'memory'), ga, ge):
        compare('read_grad_'+label, a, e, atol=1e-3, rtol=4e-4)


def geometry(q, dim):
    _, grouped = _grouped_planes(q, dim)
    table = torch.as_tensor(grouped.reshape(-1, grouped.shape[-1]), device='cuda', dtype=torch.int32)
    matrix = torch.zeros(table.shape[0], q**dim, device='cuda')
    matrix.scatter_(1, table.long(), 1)
    transpose = matrix.T.nonzero()[:, 1].reshape(q**dim, -1).contiguous().int()
    return table, transpose, matrix


def check_top4(n):
    x = torch.randn(2, 37, 3, n, device='cuda').permute(0, 2, 1, 3).requires_grad_()
    bias = torch.randn(3, n, device='cuda', requires_grad=True)
    ie, scores = None, x+bias[None, :, None, :]
    values, ie = scores.flatten(0, 1).topk(4, dim=-1)
    we = values.softmax(-1)
    ia, wa = top4_softmax(x, bias)
    compare('top4_indices', ia, ie, atol=0, rtol=0)
    compare('top4_weights', wa, we, atol=1e-6, rtol=1e-6)
    dw = torch.randn_like(wa)
    ga = torch.autograd.grad(wa, (x, bias), dw)
    ge = torch.autograd.grad(we, (x, bias), dw)
    for a, e in zip(ga, ge):
        compare('top4_gradient', a, e, atol=1e-6, rtol=1e-6)


def check_incidence(q, dim):
    table, transpose, matrix = geometry(q, dim)
    f = torch.randn(2, q**dim, 17, 23, device='cuda', requires_grad=True)
    expected = torch.einsum('en,bnrp->berp', matrix.double(), f.double()).float()
    actual = incidence(f, table, transpose)
    dy = torch.randn_like(actual)
    compare('incidence_output', actual, expected)
    compare('incidence_gradient', torch.autograd.grad(actual, f, dy)[0],
            torch.autograd.grad(expected, f, dy)[0])


def check_mixer(d):
    from zoo_smat_gdn import SmatGDNReset
    ref = SmatGDNReset(64, d=d, headdim=16, n_heads=2, expand_v=1,
                      prebuild_lengths=(256,), memory_update='delta',
                      memory_key_mode='tied_causal', memory_read_k=4).cuda()
    new = copy.deepcopy(ref)
    import content_addr
    import inspect
    assert 'read_topk' in inspect.getsource(content_addr.ContentAssign.forward), 'Wrong ContentAssign module deployed'
    for g in new.ca.values():
        for ca in g.values():
            ca.read_backend = 'triton'
    x = torch.randn(2, 256, 64, device='cuda', requires_grad=True)
    z = x.detach().clone().requires_grad_()
    a, e = new(z), ref(x)
    compare(f'mixer_d{d}', a, e)
    dy = torch.randn_like(a)
    (a*dy).sum().add(new.get_auxiliary_loss()).backward()
    (e*dy).sum().add(ref.get_auxiliary_loss()).backward()
    compare(f'mixer_input_d{d}', z.grad, x.grad, atol=1e-3, rtol=1e-3)
    for (name, pn), (_, pr) in zip(new.named_parameters(), ref.named_parameters()):
        if pr.grad is not None:
            compare(f'mixer_d{d}_gradient_{name}', pn.grad, pr.grad, atol=2e-3, rtol=2e-3)


def bench(n):
    b, m, r, p = 4, 8192, 64, 64
    q = torch.randn(b, m, r, device='cuda', requires_grad=True)
    w = torch.randn(b, m, 4, device='cuda', requires_grad=True)
    f = torch.randn(b, n, r, p, device='cuda', requires_grad=True)
    idx = torch.randint(n, (b, m, 4), device='cuda')
    dy = torch.randn(b, m, p, device='cuda')
    results = {}
    for name, fn in [('torch', read_sorted), ('triton', read_topk)]:
        def step():
            torch.autograd.grad(fn(q, idx, w, f), (q, w, f), dy)
        step()
        results[name] = triton.testing.do_bench(step, warmup=100, rep=500)
    print('READ_BENCH '+json.dumps(dict(n=n, milliseconds=results,
                                       speedup=results['torch']/results['triton'])), flush=True)


if __name__ == '__main__':
    torch.manual_seed(123)
    torch.set_num_threads(4)
    for n in (13, 97, 552, 1463):
        check_top4(n)
    for shape in [(2, 67, 13, 17, 23), (2, 129, 49, 64, 64), (1, 97, 13, 16, 16)]:
        check_read(*shape)
    check_read(2, 67, 13, 17, 23, hot=True)
    check_incidence(7, 2)
    check_incidence(11, 3)
    for d in (2, 3, 4):
        check_mixer(d)
    print('GPU_CORRECTNESS_PASSED', flush=True)
    for n in (97, 552, 1463):
        bench(n)

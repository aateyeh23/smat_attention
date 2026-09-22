"""Compare aggregation implementations at the actual 16K training shapes."""
import json
import torch
import triton
from test_smat_read_gpu import geometry, compare
from smat_aggregation_experiments import incidence, dense_incidence, read_projection


torch.manual_seed(123)
torch.set_num_threads(4)
for q, dim in ((23, 2), (11, 3)):
    table, transpose, matrix = geometry(q, dim)
    x = torch.randn(4, q**dim, 64, 64, device='cuda', requires_grad=True)
    ref = lambda x: torch.einsum('en,bnrp->berp', matrix, x)
    sparse = lambda x: incidence(x, table, transpose)
    dense = lambda x: dense_incidence(x, matrix)
    y = ref(x)
    dy = torch.randn_like(y)
    gx = torch.autograd.grad(y, x, dy)[0]
    results = {}
    for name, fn in (('torch', ref), ('sparse', sparse), ('tensor_core', dense)):
        actual = fn(x)
        compare('incidence_'+name, actual, y, atol=1e-4, rtol=1e-4)
        compare('incidence_gradient_'+name, torch.autograd.grad(actual, x, dy)[0], gx,
                atol=1e-4, rtol=1e-4)
        def step():
            torch.autograd.grad(fn(x), x, dy)
        results[name] = triton.testing.do_bench(step, warmup=100, rep=500)
    print('INCIDENCE_BENCH '+json.dumps(dict(q=q, dim=dim, milliseconds=results)), flush=True)

for b, length, width, h, n in ((2, 37, 67, 3, 41), (2, 8192, 1024, 2, 1463)):
    x = torch.randn(b, length, width, device='cuda', requires_grad=True)
    w = (torch.randn(h, n, width, device='cuda')/width**0.5).requires_grad_()
    reference = lambda: torch.einsum('blm,hem->bhle', x, w)
    ref = reference()
    actual = read_projection(x, w)
    dy = torch.randn_like(actual)
    compare('projection', actual, ref, atol=2e-4, rtol=2e-4)
    ga = torch.autograd.grad(actual, (x, w), dy)
    ge = torch.autograd.grad(ref, (x, w), dy)
    for a, e in zip(ga, ge):
        compare('projection_gradient', a, e, atol=2e-3, rtol=2e-3)
    timings = {}
    for name, fn in (('torch', reference), ('triton', lambda: read_projection(x, w))):
        timings[name] = triton.testing.do_bench(lambda: torch.autograd.grad(fn(), (x, w), dy), warmup=100, rep=500)
    print('PROJECTION_BENCH '+json.dumps(dict(shape=[b,length,width,h,n], milliseconds=timings)), flush=True)

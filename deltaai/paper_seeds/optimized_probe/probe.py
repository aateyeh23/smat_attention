"""Isolated upstream operator correctness and steady-state timing probe."""
import argparse
import json
import time
from pathlib import Path
import torch
import joint_loglinear as reference

parser = argparse.ArgumentParser()
parser.add_argument('--family', choices=['mamba2', 'gdn'], required=True)
parser.add_argument('--backend', choices=['upstream','tree','tree64'],default='upstream')
args = parser.parse_args()
import importlib
optimized=importlib.import_module({'upstream':'upstream_adapter','tree':'tree_operator','tree64':'tree64_operator'}[args.backend])
torch.manual_seed(321)
torch.backends.cuda.matmul.allow_tf32 = False
records = []
for length in [64, 100, 256, 772, 1540, 3076]:
    shape = (2, length, 2, 16)
    q = (torch.randn(shape, device='cuda') * .2).bfloat16().requires_grad_()
    k = (torch.randn_like(q) * .2).requires_grad_()
    v = torch.randn(shape,device='cuda',dtype=torch.float32 if args.family=='mamba2' else torch.bfloat16).requires_grad_()
    g = (-torch.rand(shape[:3], device='cuda') * .1).requires_grad_()
    lam = (torch.rand((*shape[:3], 13), device='cuda') + .1).bfloat16().requires_grad_()
    beta = torch.rand(shape[:3], device='cuda', dtype=torch.bfloat16).requires_grad_() if args.family == 'gdn' else None
    variables = [q,k,v,g,lam] + ([] if beta is None else [beta])
    expected = reference.log_linear(q,k,v,g,lam,beta)
    actual = optimized.log_linear(q,k,v,g,lam,beta)
    torch.cuda.synchronize()
    err = ((actual.float()-expected.float()).norm()/expected.float().norm()).item()
    print('FORWARD',args.family,length,err,flush=True)
    dy = torch.randn_like(actual)
    ga = torch.autograd.grad(actual,variables,dy)
    ge = torch.autograd.grad(expected,variables,dy)
    errs = [((a.float()-e.float()).norm()/e.float().norm().clamp_min(1e-8)).item() for a,e in zip(ga,ge)]
    record = dict(family=args.family,length=length,forward_error=err,gradient_errors=errs)
    records.append(record)
    print('CHECK',json.dumps(record),flush=True)
    assert err < .025 and max(errs) < .06, record
Path(__file__).with_name(f'checks-{args.backend}-{args.family}.json').write_text(json.dumps(records,indent=2)+'\n')
print('ALL_OPERATOR_CHECKS_PASSED',args.family,flush=True)

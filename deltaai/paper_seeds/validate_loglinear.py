"""Validate upstream long-sequence operator before admitting production runs."""
import json
from pathlib import Path
import time
import torch
from joint_loglinear import log_linear, JointLogLinearMixer
from zoo_log_linear import log_linear as dense

torch.manual_seed(321)
torch.backends.cuda.matmul.allow_tf32 = False
records = []
for family in ['mamba2', 'gdn']:
    for length in [64, 100, 256, 772, 1540, 3076]:
        shape = (2, length, 2, 16)
        q = (torch.randn(shape, device='cuda')*.2).bfloat16().requires_grad_()
        k = (torch.randn_like(q)*.2).requires_grad_()
        v = torch.randn_like(q).requires_grad_()
        g = (-torch.rand(shape[:3], device='cuda')*.1).requires_grad_()
        lam = (torch.rand((*shape[:3], 13), device='cuda')+.1).bfloat16().requires_grad_()
        beta = torch.rand(shape[:3], device='cuda', dtype=torch.bfloat16).requires_grad_() if family=='gdn' else None
        actual = log_linear(q,k,v,g,lam,beta)
        qf,kf=q.float(),k.float()
        if beta is not None:
            qf=qf*torch.rsqrt(qf.square().sum(-1,keepdim=True)+1e-6)
            kf=kf*torch.rsqrt(kf.square().sum(-1,keepdim=True)+1e-6)
        expected = dense(qf,kf,v.float(),g,lam.float(),None if beta is None else beta.float())
        error=(actual.float()-expected).norm()/expected.norm()
        assert error < .025, (family,length,error.item())
        variables=[q,k,v,g,lam]+([] if beta is None else [beta])
        dy=torch.randn_like(actual)
        ga=torch.autograd.grad(actual,variables,dy)
        ge=torch.autograd.grad(expected,variables,dy.float())
        gradient_errors=[]
        for a,e in zip(ga,ge):
            err=(a.float()-e.float()).norm()/e.float().norm().clamp_min(1e-8)
            assert err < .06,(family,length,err.item())
            gradient_errors.append(err.item())
        record=dict(family=family,length=length,relative_forward_error=error.item(),relative_gradient_errors=gradient_errors)
        records.append(record);print('OPERATOR_PASS',json.dumps(record),flush=True)
    model=JointLogLinearMixer(64,family=family).cuda()
    for length in [64,100,772,1540,3076]:
        model.zero_grad(set_to_none=True)
        x=torch.randn(256,length,64,device='cuda',requires_grad=True)
        torch.cuda.synchronize();started=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y=model(x);loss=y.float().square().mean()
        loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print('FULL_BATCH_PASS',family,length,time.monotonic()-started,flush=True)
        del x,y,loss
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        x=torch.randn(2,100,64,device='cuda');y=model(x);x[:,50:]=torch.randn_like(x[:,50:])
        torch.testing.assert_close(model(x)[:,:50],y[:,:50],atol=.002,rtol=.02)
root=Path(__file__).resolve().parent.parent/'results/paper_seeds_20260920'
(root/'loglinear_validated.json').write_text(json.dumps(records,indent=2)+'\n')
print('ALL_CHECKS_PASSED',flush=True)

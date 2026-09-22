"""Gate training on independent recurrence, causality, gradients and batch256."""
import json
import time
import numpy as np
from common import ROOT,DATA,LENGTHS,bootstrap,verify_sources,verify_data,atomic_json

def main():
    bootstrap()
    import torch
    from torch.nn import functional as F
    import joint_recall
    from synthetic_memory import make_model,query_logits,auxiliary
    from mom_model import MatchedMom,apply_ablation
    torch.set_num_threads(2)
    digest=verify_sources();data_digest=verify_data()
    torch.manual_seed(919)
    m=MatchedMom({16:8}).cuda()
    x=torch.randn(2,16,64,device='cuda',requires_grad=True)
    selected=torch.stack([torch.stack([torch.randperm(8,device='cuda')[:4] for _ in range(16)]) for _ in range(2)])
    weights=torch.randn(2,16,4,device='cuda').softmax(-1)
    with torch.autocast('cuda',dtype=torch.bfloat16): actual=m._routed(x,selected,weights,8)
    # Separate eager implementation: explicit stream extraction, causal conv,
    # FP32 delta recurrence and direct token-wise weighted reconstruction.
    reference=x.new_zeros(2,16,2,16)
    for batch in range(2):
        for memory in range(8):
            times,slots=torch.where(selected[batch]==memory)
            if times.numel()==0:continue
            source=x[batch,times]
            proj=F.linear(source,m.expert_weight[memory])
            k,v,beta,g=proj.split([32,32,2,2],-1)
            q=m.q_proj(source)
            convolved=[]
            for z,module in ((q,m.q_conv1d),(k,m.k_conv1d),(v,m.v_conv1d)):
                z=F.silu(F.conv1d(F.pad(z.T[None],(3,0)),module.weight,
                    bias=module.bias,groups=32))[0].T.reshape(-1,2,16)
                convolved.append(z)
            q,k,v=convolved
            q=F.normalize(q,dim=-1);k=F.normalize(k,dim=-1)
            g=-m.A_log.exp()*F.softplus(g+m.dt_bias)
            state=x.new_zeros(2,16,16)
            outputs=[]
            for t in range(len(times)):
                state=state*g[t].exp()[:,None,None]
                residual=v[t]-torch.einsum('hk,hkv->hv',k[t],state)
                state=state+beta[t].sigmoid()[:,None,None]*k[t,:, :,None]*residual[:,None,:]
                outputs.append(torch.einsum('hk,hkv->hv',q[t],state)*.25)
            reference[batch,times]=reference[batch,times]+torch.stack(outputs)*weights[batch,times,slots,None,None]
    relative=float((actual.float()-reference).norm()/reference.norm())
    assert relative<.03,relative
    probe=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*probe).sum(),(x,m.expert_weight),retain_graph=True)
    gr=torch.autograd.grad((reference*probe.float()).sum(),(x,m.expert_weight))
    gradient_errors=[float((a-r).norm()/r.norm()) for a,r in zip(ga,gr)]
    assert max(gradient_errors)<.06,gradient_errors
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        m.train();before=m(x.detach());m.get_auxiliary_loss()
        changed=x.detach().clone();changed[:,8:]=torch.randn_like(changed[:,8:])*2
        after=m(changed);m.get_auxiliary_loss()
        torch.testing.assert_close(before[:,:8],after[:,:8],atol=.002,rtol=.02)
        changed=x.detach().clone();changed[1]=torch.randn_like(changed[1])*2
        separate=m(changed);m.get_auxiliary_loss()
        torch.testing.assert_close(before[0],separate[0],atol=.002,rtol=.02)
        m.eval();evaluation=m(x.detach());m.get_auxiliary_loss()
        torch.testing.assert_close(before,evaluation,atol=0,rtol=0)
    print('PASS independent recurrence, backward, future-token causality, batch isolation and train/eval agreement',flush=True)
    del m,x,actual,reference
    joint_recall.configure_dataset(DATA)
    data=joint_recall.load_data(DATA,'shared','train')
    audits=[]
    for variant in ('mom_profiles','mom_bytes'):
        torch.manual_seed(123)
        model=make_model('gdn_current',64,3,LENGTHS,joint_recall.VOCAB)
        audit=apply_ablation(model,variant)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.1)
        def step(arrays):
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,targets=query_logits(model,arrays)
                loss=F.cross_entropy(logits.flatten(0,1).float(),targets.flatten())+auxiliary(model)
            assert torch.isfinite(loss)
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None:assert torch.isfinite(p.grad).all(),name
            for mixer in (m for m in model.modules() if isinstance(m,MatchedMom)):
                assert mixer.expert_weight.grad is not None and mixer.expert_weight.grad.norm()>0
                assert mixer.routers[str(arrays[0].shape[1])].weight.grad.norm()>0
            optimizer.step()
            return float(loss.detach())
        integration=[]
        for arrays in data:
            batch=joint_recall.batch(arrays,np.arange(2))
            loss=step(batch)
            integration.append(dict(length=int(batch[0].shape[1]),loss=loss))
            print('PASS',variant,'length',batch[0].shape[1],'loss',loss,flush=True)
        batch=joint_recall.batch(data[-1],np.arange(256))
        torch.cuda.reset_peak_memory_stats();timings=[]
        for _ in range(3):
            torch.cuda.synchronize();started=time.perf_counter();loss=step(batch)
            torch.cuda.synchronize();timings.append(time.perf_counter()-started)
            print('FULL_BATCH',variant,'seconds',timings[-1],'loss',loss,flush=True)
        audit.update(integration=integration,full_batch_step_seconds=float(np.median(timings[1:])),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        audits.append(audit)
        del model,optimizer
        torch.cuda.empty_cache()
    atomic_json(ROOT/'validated.json',dict(passed=True,source_manifest_sha256=digest,
        dataset_manifest_sha256=data_digest,reference_relative_error=relative,
        reference_gradient_errors=gradient_errors,causality=True,batch_isolation=True,
        train_eval_equal=True,audits=audits,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__))
    print('ALL_VALIDATIONS_PASSED',flush=True)

if __name__=='__main__':main()

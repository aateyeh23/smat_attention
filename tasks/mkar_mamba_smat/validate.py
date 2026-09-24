"""Verify original task semantics, unchanged softmax, and every learned mixer."""
import copy
import importlib.util
from pathlib import Path
from unittest.mock import patch
import gc
import hashlib
import json
import os
import time
import numpy as np
from common import ROOT,PROTOCOL as P,CONFIGS,bootstrap,verify_sources,atomic_json

bootstrap()
import torch
from mkar import MKARModel as OriginalModel,evaluate
from campaign_model import Model
from learned_mixers import make_mixer
from train import task_for,loss_for
from smat.mask import build_mask
from smat.attention import to_device


def data_checks():
    audits=[]
    for k in (1,2,3):
        a,b=task_for(k,123,'cpu'),task_for(k,123,'cpu')
        aa,bb=a.batch(4),b.batch(4)
        assert all(torch.equal(x,y) for x,y in zip(aa,bb))
        x,pay,tgt,rows,req,positions=aa
        assert x.shape==(4,1024) and pay.shape==(4,1024,8)
        assert tgt.shape==(4,8,8) and torch.all(tgt.sum(-1)==k)
        assert torch.all(x>0) and torch.any(x>=a.NOISE0)
        for i in range(4):
            marked=(pay[i].sum(-1)>0).nonzero().flatten()
            assert len(marked)==8 and torch.all(marked<512)
            assert torch.all(x[i,marked]==a.PAY)
            for j in range(8):
                keys=x[i,rows[i,j]-k+1:rows[i,j]+1]
                assert set(keys.tolist())==set(req[i,j].tolist())
                assert torch.equal(x[i,positions[i,j]-1],req[i,j])
                assert torch.equal(pay[i,positions[i,j]].sum(0),tgt[i,j])
        state=copy.deepcopy(a.rng.bit_generator.state)
        expected=a.batch(4);a.rng.bit_generator.state=state
        assert all(torch.equal(x,y) for x,y in zip(expected,a.batch(4)))
        audits.append(dict(k=k,batch_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            noisy_inputs=True,separate_key_payload=True,serialized_requests=True,oracle_targets_passed=True,
            generator_resume_passed=True))
    return audits


def main():
    torch.set_num_threads(2)
    source_hash=verify_sources()
    datasets=data_checks()
    results=[];common=None;pair_parameters={}
    for config in CONFIGS:
        torch.manual_seed(123)
        model=Model(config,123).cuda()
        shared={n:p.detach().cpu().clone() for n,p in model.named_parameters() if '.attn.' not in n}
        if common is None:common=shared
        else:assert shared.keys()==common.keys() and all(torch.equal(v,common[n]) for n,v in shared.items())
        if config['d']==3:
            params={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
            if config['random_incidence']:
                assert all(torch.equal(v,pair_parameters[config['family']][n]) for n,v in params.items())
            else:pair_parameters[config['family']]=params
        original=Path(os.environ.get('MKAR_PARENT_MIXERS','code/results/learned_mkar_original_mamba_transport_normalized_20260923/source/experiment/learned_mixers.py'))
        if original.exists():
            spec=importlib.util.spec_from_file_location('original_mamba_factory',original)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            factory=mod.make_mixer
        else:
            # The parent campaign (incidence normalization without the writer-feature
            # stop-gradient) is not in this repository: its factory is this one with
            # that single switch off.
            factory=lambda cfg,layer:make_mixer({**cfg,'detach_write_hash_features':False},layer)
        with patch('campaign_model.make_mixer',factory):reference=Model(config,123).cuda()
        assert model.blocks[0].attn.memory_incidence_rescale and reference.blocks[0].attn.memory_incidence_rescale
        assert model.state_dict().keys()==reference.state_dict().keys()
        assert all(torch.equal(v,model.state_dict()[n]) for n,v in reference.state_dict().items())
        assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in reference.parameters())
        gradient_isolation=[]
        for k in (1,2,3):
            data=task_for(k,991).batch(2)
            outputs=[]
            for name,m in [('detached',model),('parent',reference)]:
                captured={}
                def hook(module,args):captured['input']=args[0]
                handle=m.blocks[0].attn.register_forward_pre_hook(hook)
                outputs.append(m(*data[:2]))
                mixer=m.blocks[0].attn
                ca=next(iter(mixer.ca['1024'].values()))
                aux=m.auxiliary()
                gu,gconv,ghash=torch.autograd.grad(aux,[captured['input'],mixer.kconv.weight,ca.W],allow_unused=True)
                magnitude=0. if gu is None else float(gu.abs().max())
                assert magnitude==0.
                assert torch.isfinite(ghash).all() and ghash.abs().max()>0
                conv_magnitude=0. if gconv is None else float(gconv.abs().max())
                if name=='detached':assert conv_magnitude==0.
                else:assert torch.isfinite(gconv).all() and conv_magnitude>0.
                gradient_isolation.append(dict(k=k,model=name,balance_input_gradient_max=magnitude,
                    balance_conv_gradient_max=conv_magnitude,balance_hash_gradient_max=float(ghash.abs().max())))
                handle.remove()
            torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-6,atol=1e-7)
        del reference,outputs,data
        optimizer=torch.optim.AdamW(model.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
        steps=[]
        for k in (1,2,3):
            task=task_for(k,123)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();start=time.monotonic()
            ce,loss=loss_for(model,task.batch(P['batch']))
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert model.blocks[0].attn.memory_qk.weight.grad is not None
            assert torch.isfinite(model.blocks[0].attn.memory_qk.weight.grad).all()
            assert model.blocks[0].attn.memory_qk.weight.grad.abs().max()>0
            assert model.blocks[0].attn.kconv.weight.grad is not None
            assert torch.isfinite(model.blocks[0].attn.kconv.weight.grad).all()
            assert model.blocks[0].attn.kconv.weight.grad.abs().max()>0
            grads=[]
            if config['d']>=2:
                for block in model.blocks:
                    for modules in block.attn.ca.values():
                        for ca in modules.values():
                            assert ca.W_read.grad is not None and ca.W.grad is not None
                            assert torch.isfinite(ca.W_read.grad).all() and torch.isfinite(ca.W.grad).all()
                            grads.append(dict(writer=float(ca.W.grad.abs().max()),reader=float(ca.W_read.grad.abs().max())))
            optimizer.step();torch.cuda.synchronize()
            steps.append(dict(k=k,bce=float(ce.detach()),gradient_norm=float(norm),routing_gradients=grads,
                              seconds=time.monotonic()-start))
        # Original evaluator must accept the modern ca ModuleDict and preserve its RNG semantics.
        evaluation=evaluate(model,task_for(3,987),512,2,1,False)
        assert 0<=evaluation['exact_support']<=1 and np.isfinite(evaluation['mse'])
        record=dict(config=config['name'],passed=True,steps=steps,evaluation=evaluation,audit=model.audit(),gradient_isolation=gradient_isolation)
        results.append(record);atomic_json(ROOT/'checks/integration_progress.json',results)
        print('PASS',json.dumps(record),flush=True)
        if config['family']=='mamba2':
            # Actual weights, optimizer and generator continuation reproduce one next update.
            task=task_for(2,9101)
            checkpoint=ROOT/'checks/resume.pt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=task.rng.bit_generator.state),checkpoint)
            restored=Model(config,123).cuda();other=torch.optim.AdamW(restored.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
            ck=torch.load(checkpoint,weights_only=False);restored.load_state_dict(ck['model']);other.load_state_dict(ck['optimizer'])
            other_task=task_for(2,0);other_task.rng.bit_generator.state=ck['rng']
            for m,opt,t in ((model,optimizer,task),(restored,other,other_task)):
                opt.zero_grad(set_to_none=True);_,loss=loss_for(m,t.batch(2));loss.backward();opt.step()
            for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b)
            checkpoint.unlink();del restored,other,ck
        del model,optimizer,ce,loss
        gc.collect();torch.cuda.empty_cache()
    atomic_json(ROOT/'validated.json',dict(passed=True,source_manifest_sha256=source_hash,
        original_data_checks=datasets,configs=results,identical_state_and_full_forward_passed=True, balance_input_and_key_conv_gradient_isolation_passed=True,
        common_initialization_passed=True,checkpoint_and_rng_resume_passed=True,
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__))
    print('ALL_VALIDATIONS_PASSED',flush=True)


if __name__=='__main__':main()

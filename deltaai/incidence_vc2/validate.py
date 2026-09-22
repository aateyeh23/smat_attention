"""Independently gate the VC-2 runs on exact combinatorics and GPU integration."""
import dataclasses
import importlib
import json
import sys
import time
import numpy as np
from common import ROOT,SOURCE,DATA,LENGTHS,bootstrap,verify_sources,verify_data,atomic_json


def main():
    bootstrap()
    import torch
    from torch.nn import functional as F
    import joint_recall
    from synthetic_memory import make_model,query_logits,auxiliary
    from vc2_model import apply_ablation
    from generate import has_shattered_triple
    from vc_dimension import pseudo_dimension_detailed
    torch.set_num_threads(2)
    digest=verify_sources();data_digest=verify_data()
    imports={}
    for name in ['joint_recall','synthetic_memory','content_addr','zoo_gdn_transport',
                 'zoo_smat_gdn','smat_pool_ops','smat_write_hash_grad','zoology.model']:
        path=str(importlib.import_module(name).__file__)
        assert path.startswith(str(SOURCE)+'/'),(name,path)
        imports[name]=path
    certificates=[]
    with np.load(ROOT/'incidence.npz') as matrices:
        for key in matrices.files:
            m=matrices[key];q=m.shape[1];c=m.reshape(-1,m.shape[-1])
            assert np.all(m.sum(-1)==q) and np.all(m.sum(1)==1)
            assert np.all(m.sum((0,1))==q+1)
            assert len(np.unique(c,axis=0))==q*(q+1)
            result=pseudo_dimension_detailed(c,time_limit=30.)
            assert result.exact and result.dimension==2 and result.witness.verify(c)
            assert not has_shattered_triple(m)
            gram=c.T@c
            assert np.any(gram[np.triu_indices(q*q,1)]!=1)
            certificates.append(dict(key=key,vc=2,exact=True,no_shattered_triples=True,
                                     not_relabeling_of_affine_plane=True,witness=dataclasses.asdict(result.witness)))
    print('PASS exact VC2, unique summaries, matched degrees, and non-geometric pair intersections for all 10 matrices',flush=True)
    joint_recall.configure_dataset(DATA)
    data=joint_recall.load_data(DATA,'shared','train')
    torch.manual_seed(123)
    model=make_model('gdn_current',64,3,LENGTHS,joint_recall.VOCAB)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    audit=apply_ablation(model,'random_vc2',17)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    for n,p in model.named_parameters():assert torch.equal(p,before[n]),n
    assert audit['trainable_parameters']==193064 and audit['parameters']==209960
    del before
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.1)
    integration=[]
    for arrays in data:
        batch=joint_recall.batch(arrays,np.arange(2))
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,targets=query_logits(model,batch)
            loss=F.cross_entropy(logits.flatten(0,1).float(),targets.flatten())+auxiliary(model)
        assert torch.isfinite(loss)
        loss.backward()
        for n,p in model.named_parameters():
            if p.grad is not None:assert torch.isfinite(p.grad).all(),n
        for mixer in (m for m in model.modules() if m.__class__.__name__=='SmatGDNTransport'):
            ca=next(iter(mixer.ca[str(batch[0].shape[1])].values()))
            assert ca.W.grad is not None and ca.W.grad.norm()>0
            assert ca.W_read.grad is not None and ca.W_read.grad.norm()>0
            assert ca.last_write_idx.shape[-1]==1 and ca.last_read_idx.shape[-1]==4
            assert torch.all(ca.last_write_weights==1)
            assert (ca.last_read_idx.sort(-1).values.diff(dim=-1)>0).all()
        optimizer.step()
        integration.append(dict(length=int(batch[0].shape[1]),loss=float(loss.detach())))
    batch=joint_recall.batch(data[-1],np.arange(256))
    torch.cuda.reset_peak_memory_stats();timings=[]
    for _ in range(4):
        torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,targets=query_logits(model,batch)
            loss=F.cross_entropy(logits.flatten(0,1).float(),targets.flatten())+auxiliary(model)
        assert torch.isfinite(loss)
        loss.backward();optimizer.step();torch.cuda.synchronize()
        timings.append(time.perf_counter()-started)
    result=dict(passed=True,source_manifest_sha256=digest,dataset_manifest_sha256=data_digest,
        imports=imports,certificates=certificates,integration=integration,audit=audit,
        full_batch_step_seconds=float(np.median(timings[1:])),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__)
    atomic_json(ROOT/'validated.json',result)
    print('ALL_VALIDATIONS_PASSED '+json.dumps({k:v for k,v in result.items() if k not in ['audit','imports','certificates','integration']}),flush=True)


if __name__=='__main__':main()

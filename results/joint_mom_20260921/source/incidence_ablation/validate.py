"""Gate training on source/data integrity and forward/backward integration checks."""
import copy
import importlib
import json
import time
import numpy as np
from common import ROOT, SOURCE, DATA, LENGTHS, bootstrap, verify_sources, verify_data, atomic_json


def main():
    bootstrap()
    import torch
    from torch.nn import functional as F
    import joint_recall
    from content_addr import ContentAssign
    from model import RectangularBuckets, apply_ablation
    from synthetic_memory import make_model, query_logits, auxiliary
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    digest = verify_sources()
    data_digest = verify_data()
    imports = {}
    for name in ['joint_recall', 'synthetic_memory', 'content_addr', 'zoo_gdn_transport',
                 'zoo_smat_gdn', 'smat_pool_ops', 'smat_write_hash_grad', 'zoology.model']:
        filename = str(importlib.import_module(name).__file__)
        assert filename.startswith(str(SOURCE)+'/'), (name, filename)
        imports[name] = filename
    joint_recall.configure_dataset(DATA)
    data = joint_recall.load_data(DATA, 'shared', 'train')
    torch.manual_seed(123)
    reference = make_model('gdn_current', 64, 3, LENGTHS, joint_recall.VOCAB)
    reference_state = copy.deepcopy(reference.state_dict())
    reference_parameters = {n: p.detach().clone() for n,p in reference.named_parameters()}
    first = next(m for m in reference.modules() if isinstance(m, ContentAssign))
    # Generalizing to equal radices must reproduce both routing and its STE gradients.
    original = copy.deepcopy(first)
    rectangular = copy.deepcopy(first)
    original.mode = rectangular.mode = 'point'
    original.plant = rectangular.plant = False
    rectangular.__class__ = RectangularBuckets
    rectangular.radices = (first.q, first.q)
    source = torch.randn(2, 19, 64, device='cuda')
    token_weights = torch.rand(2*first.h, 19, device='cuda')
    for neighbors in (False, True):
        values = []
        for ca in (original, rectangular):
            ca.zero_grad(set_to_none=True)
            ca._cells(source, tok_w=token_weights, retain_neighbors=neighbors)
            indices, weights = ca._sparse
            coeff = torch.sin(indices.float()*.7)
            objective = (coeff*weights).sum()+ca.aux
            grads = torch.autograd.grad(objective, (ca.W, ca.gamma, ca.b))
            values.append((indices.detach().clone(), weights.detach().clone(), ca.aux.detach(), grads))
        assert torch.equal(values[0][0], values[1][0])
        for a,b in zip(values[0][1:3], values[1][1:3]):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        for a,b in zip(values[0][3],values[1][3]):
            torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    print('PASS equal-radix writer forward and surrogate gradients', flush=True)
    rows = []
    for variant, topology in [('geometry',0),('random',17),('random',29),
                              ('buckets_profiles',0),('buckets_bytes',0)]:
        torch.manual_seed(123)
        model = make_model('gdn_current',64,3,LENGTHS,joint_recall.VOCAB)
        audit = apply_ablation(model,variant,topology)
        if variant in ('geometry','random'):
            for name,p in model.named_parameters():
                assert torch.equal(p,reference_parameters[name]), name
        if variant == 'geometry':
            for name,t in model.state_dict().items():
                assert torch.equal(t,reference_state[name]), name
        if variant == 'random':
            for row in audit['matrix_tables']:
                assert row['pairwise_profile_overlap_histogram'] != {'1': row['profile_count']*(row['profile_count']-1)//2}
            # Independent dense aggregation reference and adjoint, with random memberships.
            ca = next(m for m in model.modules() if isinstance(m,ContentAssign))
            matrix = ca.M.flatten(0,1)
            memory = torch.randn(2,ca.N0,3,4,device='cuda',requires_grad=True)
            actual = torch.einsum('tx,bxrp->btrp',matrix,memory)
            expected = torch.stack([memory[:,row.bool()].sum(1) for row in matrix],1)
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
            cotangent = torch.randn_like(actual)
            ga = torch.autograd.grad((actual*cotangent).sum(),memory,retain_graph=True)[0]
            ge = torch.autograd.grad((expected*cotangent).sum(),memory)[0]
            torch.testing.assert_close(ga,ge,atol=2e-6,rtol=2e-5)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.1)
        integration = []
        for arrays in data:
            batch = joint_recall.batch(arrays,np.arange(2))
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits, targets = query_logits(model,batch)
                loss = F.cross_entropy(logits.flatten(0,1).float(),targets.flatten())+auxiliary(model)
            assert torch.isfinite(loss)
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), (variant,name)
            active = [m for m in model.modules() if m.__class__.__name__ == 'SmatGDNTransport']
            for mixer in active:
                ca = next(iter(mixer.ca[str(batch[0].shape[1])].values()))
                assert ca.W.grad is not None and ca.W.grad.norm()>0, (variant,'writer')
                assert ca.W_read.grad is not None and ca.W_read.grad.norm()>0, (variant,'reader')
                assert ca.last_write_idx.shape[-1] == 1
                assert ca.last_read_idx.shape[-1] == 4
                assert (ca.last_read_idx.sort(-1).values.diff(dim=-1)>0).all()
                assert torch.all(ca.last_write_weights == 1)
            optimizer.step()
            integration.append(dict(length=int(batch[0].shape[1]),loss=float(loss.detach())))
        # Measure the actual training batch at the largest load; no benchmark steps saved as training.
        batch = joint_recall.batch(data[-1],np.arange(256))
        timings = []
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        for iteration in range(4):
            torch.cuda.synchronize();start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,targets=query_logits(model,batch)
                loss=F.cross_entropy(logits.flatten(0,1).float(),targets.flatten())+auxiliary(model)
            assert torch.isfinite(loss)
            loss.backward();optimizer.step();torch.cuda.synchronize()
            timings.append(time.perf_counter()-start)
        row=dict(variant=variant,topology_seed=topology,integration=integration,
                 seconds_per_train_step_3076_batch256=float(np.median(timings[1:])),
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),audit=audit)
        rows.append(row)
        print('PASS '+json.dumps({k:v for k,v in row.items() if k!='audit'}),flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
    result=dict(passed=True,source_manifest_sha256=digest,dataset_manifest_sha256=data_digest,
                imports=imports,torch_version=torch.__version__,cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),checks=rows)
    atomic_json(ROOT/'validated.json',result)
    print('ALL_VALIDATIONS_PASSED',flush=True)


if __name__ == '__main__':
    main()

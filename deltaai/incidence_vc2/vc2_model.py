"""Replace only incidence with pre-certified degree-matched VC-2 matrices."""
import hashlib
import importlib.util
import json
import numpy as np
import torch
from common import CODE,ROOT


def apply_ablation(model,variant,topology=17):
    assert variant=='random_vc2' and topology==17
    with np.load(ROOT/'incidence.npz') as matrices:
        mixers=[m for m in model.modules() if m.__class__.__name__=='SmatGDNTransport']
        assert len(mixers)==2
        certificates=json.loads((ROOT/'construction.json').read_text())['matrices']
        for layer,mixer in enumerate(mixers):
            for length,modules in mixer.ca.items():
                matrix=matrices[f'l{layer}_t{length}']
                record=next(x for x in certificates if x['layer']==layer and x['length']==int(length))
                assert hashlib.sha256(matrix.tobytes()).hexdigest()==record['incidence_sha256']
                assert record['vc_exact'] and record['vc_dimension']==2
                for ca in modules.values():
                    assert tuple(ca.M.shape)==matrix.shape
                    assert np.array_equal(matrix.sum(-1),ca.M.detach().cpu().numpy().sum(-1))
                    assert np.array_equal(matrix.sum((0,1)),ca.M.detach().cpu().numpy().sum((0,1)))
                    ca.M.copy_(torch.from_numpy(matrix).to(ca.M.device))
                    ca.coset_of.copy_(ca.M.argmax(1))
    # Reuse unchanged resource accounting; the 'geometry' path is a no-op.
    spec=importlib.util.spec_from_file_location('original_incidence_model',CODE.parent/'incidence_ablation/model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    audit=module.apply_ablation(model,'geometry')
    audit.update(variant=variant,topology_seed=topology,single_summary_vc_dimension=2,
                 construction='Degree-preserving constrained random rewiring, not uniform sampling',
                 four_read_vc_dimension='Not constrained to 2')
    return audit

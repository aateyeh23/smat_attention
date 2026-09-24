"""Freeze the normalized Mamba candidate only after all pilot final scores qualify.

This is how the confirmation campaign was set up from the seed-123 pilot.  It
copies the pilot's frozen source tree, which is not in this repository, so it
documents the selection gate rather than being needed for a rerun (README.md)."""
import json
import shutil
from pathlib import Path
from common import ROOT, SOURCE, REPO, PROTOCOL, SEEDS, CONFIGS, atomic_json, digest

PARENT = REPO / 'results/mkar_mamba_smat_20260923/pilot_s123'
RULE = REPO / 'results/mkar_mamba_smat_20260923/selection_rule.json'


def main():
    assert not ROOT.exists()
    rule = json.loads(RULE.read_text())
    assert rule['results_known_at_rule'] == {} and rule['replication_seeds'] == SEEDS
    manifest = json.loads((PARENT / 'source_sha256.json').read_text())
    for name, sha in manifest.items():
        assert digest(PARENT / 'source' / name) == sha, name
    validated = json.loads((PARENT / 'validated.json').read_text())
    assert validated['passed'] and validated['source_manifest_sha256'] == digest(PARENT / 'source_sha256.json')
    parent_protocol = json.loads((PARENT / 'protocol.json').read_text())
    assert {k:v for k,v in parent_protocol.items() if k != 'seed_list'} == {k:v for k,v in PROTOCOL.items() if k != 'seed_list'}
    selection = {}
    for k in (1, 2, 3):
        path = PARENT / f'runs/mamba2-d4-normalized-route-stop-s123/k{k}/result.json'
        result = json.loads(path.read_text())
        assert result['complete'] and result['steps'] == 12000 and result['k'] == k
        assert result['source_manifest_sha256'] == digest(PARENT / 'source_sha256.json')
        assert result['evaluation']['exact_support'] >= .99, 'Pilot does not meet the recorded selection rule'
        selection[str(k)] = dict(exact_support=result['evaluation']['exact_support'], source=str(path), result_sha256=digest(path))
    ROOT.mkdir(parents=True)
    for name in ('logs', 'checks'):
        (ROOT / name).mkdir()
    shutil.copytree(PARENT / 'source', SOURCE, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, SOURCE / 'experiment' / path.name)
    hashes = {str(p.relative_to(SOURCE)):digest(p) for p in sorted(SOURCE.rglob('*.py'))}
    for name, sha in manifest.items():
        if name not in ('experiment/common.py', 'experiment/prepare.py'):
            assert hashes[name] == sha, name
    atomic_json(ROOT / 'source_sha256.json', hashes)
    atomic_json(ROOT / 'protocol.json', PROTOCOL)
    atomic_json(ROOT / 'tasks.json', [dict(index=i, name=f'mamba2-d4-normalized-route-stop-s{s}', config=CONFIGS[0],
        seed=s, requested_sizes=[1,2,3], independent_models=True) for i,s in enumerate(SEEDS)])
    atomic_json(ROOT / 'provenance.json', dict(parent_campaign=str(PARENT),
        parent_source_manifest_sha256=digest(PARENT / 'source_sha256.json'),
        selection_rule=rule, selection_rule_sha256=digest(RULE), selection_evidence=selection,
        exploratory_seed=123, replication_seeds=SEEDS, all_reported_seeds=[123,*SEEDS],
        only_change='Four additional seeds; frozen model, trainer, validation, data, loss and evaluator unchanged.',
        read_normalization='inverse incidence density before the learned memory gate',
        writer_feature_gradient_isolation=True,
        hybrid_memory_update=True, not_routing_only=True))
    (ROOT / 'README.md').write_text('''# Normalized Mamba delta-memory SMat: four confirmation seeds

Seeds456,789,2026,2027; independent k1/k2/k3 models; four jobs/twelve fits.
Seed123 selected the architecture and remains in the five-seed report.
Selection requires all three final pilot scores >=99%, specified before k3
finished. All confirmation outcomes are retained; no best-checkpoint selection.

Only seeds differ from the frozen pilot. Same original task, BCE+balance0.01,
width256/layer1, batch32, 12000 updates, LR3e-4, AdamW WD0.1 and FP32.
The local Mamba recurrence is unchanged. The distant memory uses shared learned
query/key features, causal writer features, delta transport without scalar
decay, and division of its read by mean incidence before the existing gate.
Writer-address features are detached; content retrieval still trains the
shared convolution/projection. The same setting is used for every seed.
This normalization is geometry-derived and uses no task labels or token IDs.
No additional architecture change, parameters or training steps are introduced.

Native and softmax comparisons match width and exposure, not all resources.
Report the full-global memory control alongside routed results; do not infer
VC causality or a routing-specific advantage from native-baseline gains.
''')
    print('PREPARED', ROOT, '4 jobs / 12 fits', flush=True)


if __name__ == '__main__':
    main()

"""Complete the normalization x writer-gradient-isolation factorial pilot."""
import difflib
import json
import shutil
from pathlib import Path
from common import ROOT, SOURCE, REPO, PROTOCOL, CONFIGS, atomic_json, digest

PARENT = REPO / 'deltaai/results/learned_mkar_original_mamba_transport_normalized_20260923'


def main():
    assert not ROOT.exists()
    manifest = json.loads((PARENT / 'source_sha256.json').read_text())
    for name, sha in manifest.items():
        assert digest(PARENT / 'source' / name) == sha, name
    assert json.loads((PARENT / 'protocol.json').read_text()) == PROTOCOL
    evidence = {}
    for k in (1, 2, 3):
        path = PARENT / f'runs/mamba2-d4-transport-normalized-s123/k{k}/result.json'
        result = json.loads(path.read_text())
        assert result['complete'] and result['steps'] == 12000
        assert result['source_manifest_sha256'] == digest(PARENT / 'source_sha256.json')
        evidence[str(k)] = dict(source=str(path), result_sha256=digest(path), exact_support=result['evaluation']['exact_support'])
    ROOT.mkdir(parents=True)
    for name in ('logs', 'checks'):
        (ROOT / name).mkdir()
    shutil.copytree(PARENT / 'source', SOURCE, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, SOURCE / 'experiment' / path.name)
    backbone = SOURCE / 'deltaai/zoo_smat_mixer.py'
    old = backbone.read_text()
    before = '                    mods = self.ca[str(l)]'
    after = "                    if getattr(self, 'detach_write_hash_features', False):\n                        assert ksrc is not None\n                        ksrc = ksrc.detach()\n                    mods = self.ca[str(l)]"
    assert old.count(before) == 1
    backbone.write_text(old.replace(before, after))
    (ROOT / 'backbone_changes.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True), backbone.read_text().splitlines(True), fromfile='normalized/zoo_smat_mixer.py', tofile='normalized_stop/zoo_smat_mixer.py')))
    hashes = {str(p.relative_to(SOURCE)):digest(p) for p in sorted(SOURCE.rglob('*.py'))}
    for name, sha in manifest.items():
        if not name.startswith('experiment/') and name != 'deltaai/zoo_smat_mixer.py':
            assert hashes[name] == sha, name
    assert hashes['experiment/train.py'] == manifest['experiment/train.py']
    atomic_json(ROOT / 'source_sha256.json', hashes)
    atomic_json(ROOT / 'protocol.json', PROTOCOL)
    atomic_json(ROOT / 'tasks.json', [dict(index=0, name='mamba2-d4-normalized-route-stop-s123', config=CONFIGS[0], seed=123, requested_sizes=[1,2,3], independent_models=True)])
    atomic_json(ROOT / 'provenance.json', dict(parent_campaign=str(PARENT), parent_source_manifest_sha256=digest(PARENT / 'source_sha256.json'),
        parent_final_results=evidence, change='Stop-gradient at writer-address features in the incidence-normalized hybrid.',
        extra_parameters=0, initial_forward_changed=False, gradient_flow_changed=True,
        numerical_loss_data_budget_evaluator_unchanged=True, exploratory=True,
        motivation='Complete the two-factor pilot: normalization alone final100/100/18.8477; gradient isolation alone99.8047/99.8535/0 and improved k3 noise/requested diagnostic21.9 to1.5. Test their combination without additional changes.',
        independent_confirmations_required=True, native_backbone_unchanged=True))
    (ROOT / 'README.md').write_text('''# Incidence normalization plus writer-feature gradient isolation

One seed123, separate k1/k2/k3 models, one job/three fits. This is the fourth
cell of the normalization x gradient-isolation pilot. Parent, either change
alone, and their combination are all retained, including failed results.

The only change from the normalized pilot is detach(key_source) before writer
address assignment. Routing/balance gradients cannot train the shared causal
key convolution; content retrieval still trains it and the shared projection.
Router weights retain their gradients. Forward values and parameters at
initialization match the normalized parent, which validation checks.

The existing geometry-derived inverse-incidence-density normalization remains.
Local Mamba, shared memory features, independent write gates and key-dependent
delta transitions remain unchanged. No added parameters, task-specific token
conditions, targets in forward, changed data/loss or extra updates.
Same original task/evaluator, width256/layer1, batch32, 12000 updates, LR3e-4,
AdamW WD0.1, FP32 and routing balance0.01; use only final checkpoints.

This is an exploratory hybrid memory architecture, not a routing-only result.
A selected candidate requires four independent-seed confirmations and a
matched full-global control with the same gradient-isolation setting.
''')
    print('PREPARED', ROOT, '1 job / 3 fits', flush=True)


if __name__ == '__main__':
    main()

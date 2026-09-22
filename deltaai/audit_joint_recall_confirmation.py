"""Audit fresh record-recall predictions independently of the model/generator.

A valid evaluation can have a negative result. Report validity and the positive
comparison criteria separately; never equate successful execution with a win.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def same(a, b):
    assert np.allclose(a, b, rtol=0, atol=1e-12), (a, b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folder', type=Path)
    args = p.parse_args(); root = args.folder
    r = json.loads((root/'result.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    manifest = json.loads((root/'data_manifest.json').read_text())
    assert r['complete'] and selection == r['selection']
    assert manifest['split_code'] == r['split_code'] and r['split_code'] not in [1001, 2001, 3001]
    assert manifest['examples_per_cell'] == r['examples_per_cell'] == 10000
    dataset = Path(selection['dataset_root'])
    cfg = json.loads((dataset/'dataset_config.json').read_text()); assert cfg['layout'] == 'record'
    training = json.loads((dataset/'data/manifest.json').read_text())
    assert digest(dataset/'data/manifest.json') == manifest['training_manifest_sha256'] == selection['dataset_manifest_sha256']
    for filename, expected in manifest['files'].items():
        assert digest(root/filename) == expected, filename
    for filename, expected in manifest['evaluation_source_sha256'].items():
        assert digest(root/'evaluation_source'/filename) == expected, filename
    assert digest(root.parent/'confirmation_plan.json')==selection['selection_plan_sha256']
    assert manifest['evaluation_source_sha256']['run_joint_recall_confirmation_when_ready.py']==selection['selection_script_sha256']
    models=selection['models']+selection.get('additional_models',[])
    for chosen in models:
        assert digest(Path(chosen['folder'])/'checkpoint.pt') == chosen['checkpoint_sha256']
        assert digest(Path(chosen['folder'])/'recipe.json') == chosen['recipe_sha256']
        assert digest(Path(chosen['folder'])/'result.json') == chosen['training_result_sha256']
    scores, controls = {}, {}
    for cell in training['cells']:
        name, count = cell['name'], cell['records']
        contexts, keys = cell['contexts'], cell['keys']
        value_base = training['vocab']-training['values']
        x = np.load(root/f'fresh-{name}-inputs.npy', mmap_mode='r')
        y = np.load(root/f'fresh-{name}-targets.npy')
        positions = np.load(root/f'fresh-{name}-positions.npy')
        n, length = x.shape; assert n == 10000 and length == cell['length']
        assert y.shape == (n, count)
        same(positions, np.broadcast_to(length//2+2+3*np.arange(count), y.shape))
        hashes = set(); key, context = [], []
        for i, row in enumerate(x):
            info = row[1:1+3*count].reshape(count, 3)
            queries = row[length//2+1:length//2+1+3*count].reshape(count, 3)
            assert ((info[:,0]>=3)&(info[:,0]<35)).all()
            assert ((info[:,1]>=35)&(info[:,1]<value_base)).all()
            assert ((info[:,2]>=value_base)&(info[:,2]<training['vocab'])).all()
            assert row[0] == 1 and row[length//2] == 2 and not queries[:, 2].any()
            assert not row[1+3*count:length//2].any() and not row[length//2+1+3*count:].any()
            lookup = {(int(ctx), int(k)): int(v) for ctx, k, v in info}
            assert len(lookup) == count and set(lookup) == {(int(ctx), int(k)) for ctx, k, _ in queries}
            same(y[i], [lookup[int(ctx), int(k)] for ctx, k, _ in queries])
            for column, dest in [(1, key), (0, context)]:
                groups = {}
                for record in info:
                    groups.setdefault(int(record[column]), Counter())[int(record[2])] += 1
                assert len(groups)==(keys if column==1 else contexts)
                assert all(sum(freq.values())==(contexts if column==1 else keys) for freq in groups.values())
                dest.append(sum(max(freq.values()) for freq in groups.values())/count)
            h = hashlib.sha256(row.tobytes()).digest(); assert h not in hashes; hashes.add(h)
        for split in ['train', 'validation', 'test']:
            original = np.load(dataset/'data'/f'{name}-{split}-shared.npy', mmap_mode='r')
            assert all(hashlib.sha256(row.tobytes()).digest() not in hashes for row in original)
        controls[name] = dict(key_only=np.array(key), context_only=np.array(context), best_single_field=np.maximum(key, context))
        saved = np.load(root/f'oracles-{name}.npz')
        for field, values in controls[name].items(): same(values, saved[field])
        for chosen in models:
            label = chosen.get('label',f'{chosen["family"]}-d{chosen["d"]}')
            saved = np.load(root/label/f'test-{name}.npz')
            assert np.array_equal(saved['targets'], y) and np.array_equal(saved['positions'], positions)
            assert saved['predictions'].shape == y.shape
            good = saved['predictions'] == y
            same(good.mean(), r['metrics'][label]['cells'][name]['accuracy'])
            same(good.all(1).mean(), r['metrics'][label]['cells'][name]['exact_accuracy'])
            scores[label, name] = good.mean(1)
        print('AUDITED_CELL', name, flush=True)
    cells = [c['name'] for c in training['cells']]
    largest = max(training['cells'], key=lambda c: c['records'])['name']
    rng = np.random.default_rng(20260921); criteria = []
    for family in ['mamba2', 'gdn_current']:
        for variant in sorted({m['d'] for m in selection['models']} - {1}):
            draws = {k: [] for k in ['smat_minus_native', 'smat_minus_key_only', 'smat_minus_single_field']}
            differences = {k: [] for k in draws}
            for cell in cells:
                row = next(a for a in r['comparisons'] if a['family'] == family and a.get('d',3) == variant and a['cell'] == cell)
                native, smat = scores[f'{family}-d1', cell], scores[f'{family}-d{variant}', cell]
                same(row['native_accuracy'], native.mean()); same(row['smat_accuracy'], smat.mean())
                for field, metric in [('key_only', 'key_only_majority_accuracy'), ('context_only', 'context_only_majority_accuracy'), ('best_single_field', 'best_single_field_accuracy')]:
                    same(row[metric], controls[cell][field].mean())
                for field, other in [('smat_minus_native', native), ('smat_minus_key_only', controls[cell]['key_only']), ('smat_minus_single_field', controls[cell]['best_single_field'])]:
                    delta = smat-other
                    samples = np.concatenate([delta[rng.integers(0, n, (100, n))].mean(1) for _ in range(20)])
                    ci = np.quantile(samples, [.025, .975])
                    same(row[field]['difference'], delta.mean()); same(row[field]['ci95'], ci)
                    draws[field].append(samples); differences[field].append(delta.mean())
                    if cell == largest and field != 'smat_minus_key_only':
                        criteria.append(dict(family=family, d=variant, scope=cell, comparison=field, passed=bool(ci[0] > 0)))
            macro = next(a for a in r['macro_comparisons'] if a['family'] == family and a.get('d',3) == variant)
            for field in draws:
                ci = np.quantile(np.mean(draws[field], axis=0), [.025, .975])
                same(macro[field]['difference'], np.mean(differences[field])); same(macro[field]['ci95'], ci)
                if field == 'smat_minus_native':
                    criteria.append(dict(family=family, d=variant, scope='macro', comparison=field, passed=bool(ci[0] > 0)))
            for d, key in [(1, 'native_accuracy'), (variant, 'smat_accuracy')]:
                label = f'{family}-d{d}'; mean = np.mean([scores[label, cell].mean() for cell in cells])
                same(r['metrics'][label]['accuracy'], mean); same(macro[key], mean)
                same(r['metrics'][label]['exact_accuracy'], np.mean([r['metrics'][label]['cells'][cell]['exact_accuracy'] for cell in cells]))
    for chosen in selection.get('additional_models',[]):
        family=chosen['family'];label=chosen['label']
        for variant in sorted({m['d'] for m in selection['models']} - {1}):
            draws=[];differences=[]
            for cell in cells:
                row=next(a for a in r['additional_comparisons'] if a['native_label']==label and a['d']==variant and a['cell']==cell)
                native,smat=scores[label,cell],scores[f'{family}-d{variant}',cell]
                same(row['native_accuracy'],native.mean());same(row['smat_accuracy'],smat.mean())
                delta=smat-native
                samples=np.concatenate([delta[rng.integers(0,n,(100,n))].mean(1) for _ in range(20)])
                ci=np.quantile(samples,[.025,.975])
                same(row['smat_minus_native']['difference'],delta.mean());same(row['smat_minus_native']['ci95'],ci)
                draws.append(samples);differences.append(delta.mean())
                if cell==largest:
                    criteria.append(dict(family=family,d=variant,scope=cell,comparison=label,passed=bool(ci[0]>0)))
            macro=next(a for a in r['additional_macro_comparisons'] if a['native_label']==label and a['d']==variant)
            ci=np.quantile(np.mean(draws,axis=0),[.025,.975])
            same(macro['smat_minus_native']['difference'],np.mean(differences))
            same(macro['smat_minus_native']['ci95'],ci)
            same(macro['native_accuracy'],np.mean([scores[label,c].mean() for c in cells]))
            same(macro['smat_accuracy'],np.mean([scores[f'{family}-d{variant}',c].mean() for c in cells]))
            criteria.append(dict(family=family,d=variant,scope='macro',comparison=label,passed=bool(ci[0]>0)))
        same(r['metrics'][label]['accuracy'],np.mean([scores[label,c].mean() for c in cells]))
        same(r['metrics'][label]['exact_accuracy'],np.mean([r['metrics'][label]['cells'][c]['exact_accuracy'] for c in cells]))
    out = dict(passed=True, positive_result=all(c['passed'] for c in criteria if c['d']==3),
               positive_by_d={d:all(c['passed'] for c in criteria if c['d']==d) for d in sorted({c['d'] for c in criteria})},
               criteria=criteria,
               selection_sha256=digest(root/'selection.json'), result_sha256=digest(root/'result.json'))
    (root/'audit.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__': main()

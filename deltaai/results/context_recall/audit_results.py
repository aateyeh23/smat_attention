"""Recompute final metrics and verify the complete paired experimental grid (CPU)."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from context_recall import ContextRecall


def digest(data):
    return hashlib.sha256(b''.join(a.tobytes() for a in data)).hexdigest()


groups = defaultdict(list)
for path in sorted(ROOT.glob('*/result.json')):
    recipe = json.loads((path.parent/'recipe.json').read_text())
    result = json.loads(path.read_text())
    groups[(recipe['load'], recipe['shared'], recipe['seed'])].append((path.parent, recipe, result))
expected = {(r, reuse, seed) for r in [4, 8] for reuse in [0., 1.]
            for seed in ([123, 456, 789] if r == 4 else [123])}
assert set(groups) == expected
data_cache, checks = {}, []
for (records, reuse, seed), runs in sorted(groups.items()):
    assert {(q['family'], q['d']) for _, q, _ in runs} == {
        (f, d) for f in ['mamba2', 'gdn_current'] for d in [1, 2, 3, 4]}
    assert len(runs) == 8
    canonical = [{k: v for k, v in q.items() if k not in ['family', 'd', 'out']} for _, q, _ in runs]
    assert all(q == canonical[0] for q in canonical)
    q = canonical[0]
    assert (q['steps'], q['batch'], q['eval_examples'], q['test_examples'], q['width'], q['sequence_length']) == (500, 32, 1024, 4096, 64, 64)
    task = ContextRecall(q['contexts'], reuse, q['sequence_length'])
    data = task.generate('context', records, 4096, np.random.default_rng(np.random.SeedSequence([seed, 30001, records])))
    val = task.generate('context', records, 1024, np.random.default_rng(np.random.SeedSequence([seed, 10001, records])))
    # Decode record/query tokens independently of the task's evaluation helpers.
    alternatives = np.empty_like(data[2]) if reuse else None
    for b, (tokens, positions, targets) in enumerate(zip(*data)):
        starts = np.flatnonzero((tokens[:52] >= 2) & (tokens[:52] < 18))
        table = {(int(tokens[j]), int(tokens[j+1])): int(tokens[j+2]) for j in starts}
        assert len(table) == records
        for i, (p, target) in enumerate(zip(positions, targets)):
            context, key = int(tokens[p-1]), int(tokens[p])
            assert tokens[p-2] == 1 and table[(context, key)] == target
            if reuse:
                other = [v for (c, k), v in table.items() if k == key and c != context]
                assert len(other) == 1
                alternatives[b, i] = other[0]
    for folder, q, result in runs:
        assert result['complete'] and result['steps'] == 500
        assert digest(data) == result['test_sha256']
        assert digest(val) == result['validation_sha256']
        saved = np.load(folder/'test_predictions.npz')
        assert np.array_equal(saved['targets'], data[2])
        assert np.array_equal(saved['positions'], data[1])
        pred = saved['predictions']
        assert pred.shape == (4096, 4)
        correct = pred == data[2]
        assert np.isclose(correct.mean(), result['accuracy'])
        assert np.isclose(correct.all(axis=1).mean(), result['exact_accuracy'])
        if reuse:
            wrong = pred == alternatives
            assert np.isclose(wrong.mean(), result['wrong_context_accuracy'])
            assert np.isclose(100*(correct.mean()-wrong.mean()), result['context_binding_gap_pp'])
        else:
            assert result['context_binding_gap_pp'] is None
    data_cache[(records, reuse, seed)] = data
    checks.append(dict(records=records, shared=reuse, seed=seed, runs=len(runs), test_sha256=digest(data), validation_sha256=digest(val)))
for records, _, seed in sorted(expected):
    a, b = data_cache[(records, 0., seed)], data_cache[(records, 1., seed)]
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    ax, bx = a[0].copy(), b[0].copy()
    ax[(ax >= 18) & (ax < 82)] = 18
    bx[(bx >= 18) & (bx < 82)] = 18
    assert np.array_equal(ax, bx)
output = dict(passed=True, completed_runs=sum(len(v) for v in groups.values()), groups=checks,
              checks=['complete expected grid', 'matched recipes within condition/seed',
                      'independent token decoding', 'regenerated validation/test hashes',
                      'saved accuracy/exact/context-binding metrics',
                      'reuse conditions differ only in key identities'])
(ROOT/'audit.json').write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps(output, indent=2))

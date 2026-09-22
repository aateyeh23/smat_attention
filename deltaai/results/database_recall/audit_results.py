"""Independently decode database queries and check saved final predictions (CPU)."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from database_recall import DatabaseRecall


def digest(data):
    return hashlib.sha256(b''.join(a.tobytes() for a in data)).hexdigest()


cache, checked, groups, diagnostics = {}, [], {}, []
for path in sorted(ROOT.glob('*/result.json')):
    recipe = json.loads((path.parent/'recipe.json').read_text())
    result = json.loads(path.read_text())
    records, seed = recipe['load'], recipe['seed']
    diagnostic = dict(run=path.parent.name)
    assert recipe['steps'] == result['steps'] == 500 and result['complete']
    assert recipe['batch'] == 32 and recipe['width'] == 64
    key = (records, seed, recipe['sequence_length'])
    if key not in cache:
        task = DatabaseRecall(recipe['sequence_length'])
        data = [task.generate('database', records, 4096,
                  np.random.default_rng(np.random.SeedSequence([seed, 30001, records])), heldout=bool(h)) for h in [0,1]]
        val = task.generate('database', records, 1024,
                  np.random.default_rng(np.random.SeedSequence([seed, 10001, records])))
        types = []
        for h, (x, positions, targets) in enumerate(data):
            arities = []
            for b, (tokens, pos, y) in enumerate(zip(x, positions, targets)):
                start = int(pos[0])-4
                row_starts = np.flatnonzero((tokens[:start]>=3)&(tokens[:start]<19))
                table={int(tokens[j]):tokens[j+1:j+4] for j in row_starts}
                assert len(table) == records
                for qi in range(2):
                    p = pos[qi*records:(qi+1)*records]
                    predicate = tokens[p[0]-3:p[0]]
                    assert tokens[p[0]-4] == 2
                    fields = np.flatnonzero(predicate != 1)
                    assert len(fields) in [1,2]
                    arities.append(len(fields))
                    assert set(tokens[p]) == set(table)
                    if len(fields) == 2:
                        values = predicate[fields]-np.array([19,23,27])[fields]
                        assert bool(values[0]==values[1]) == bool(h)
                    decoded = [31 if np.array_equal(table[int(tokens[j])][fields],predicate[fields]) else 32 for j in p]
                    assert np.array_equal(decoded,y[qi*records:(qi+1)*records])
            types.append(np.array(arities))
        # Paired tests differ only at the conjunction predicate and its labels.
        a,b=data
        assert np.array_equal(a[1],b[1])
        ax,bx=a[0].copy(),b[0].copy()
        for i,pos in enumerate(a[1]):
            for qi in range(2):
                p=pos[qi*records]
                if types[0][i*2+qi]==2:
                    ax[i,p-3:p]=0; bx[i,p-3:p]=0
                else: assert np.array_equal(a[2][i,qi*records:(qi+1)*records],b[2][i,qi*records:(qi+1)*records])
        assert np.array_equal(ax,bx)
        cache[key]=(data,val,types)
    data,val,types=cache[key]
    assert digest(val)==result['validation_sha256']
    for h,prefix in enumerate(['','heldout_']):
        assert digest(data[h]) == result['heldout_sha256' if h else 'test_sha256']
        saved=np.load(path.parent/('heldout_predictions.npz' if h else 'test_predictions.npz'))
        assert np.array_equal(saved['targets'],data[h][2]) and np.array_equal(saved['positions'],data[h][1])
        prediction=saved['predictions']; target=data[h][2]
        assert prediction.shape==target.shape==(4096,2*records)
        assert np.isclose((prediction==target).mean(),result[prefix+'accuracy'])
        assert np.isclose((prediction==target).all(axis=1).mean(),result[prefix+'exact_accuracy'])
        pp,tt=prediction.reshape(-1,records),target.reshape(-1,records)
        correct=pp==tt
        other=((pp[:,:,None]==tt[:,None,:]).sum(axis=2)-correct)/(records-1)
        label='heldout' if h else 'seen'
        diagnostic[label+'_object_binding_gap_pp']=100*float((correct-other).mean())
        diagnostic[label+'_predicted_yes_fraction']=float((pp==31).mean())
        diagnostic[label+'_nonconstant_query_fraction']=float((pp!=pp[:,:1]).any(axis=1).mean())
        for name,mask in [('all',np.ones(8192,dtype=bool)),('single',types[h]==1),('conjunction',types[h]==2)]:
            p=prediction.reshape(-1,records)[mask];t=target.reshape(-1,records)[mask]
            positive=t==31; correct=p==t; nonempty=positive.any(axis=1)
            ba=(correct[positive].mean()+correct[~positive].mean())/2
            assert np.isclose(ba,result[prefix+name+'_balanced_accuracy'])
            assert np.isclose(correct.all(axis=1).mean(),result[prefix+name+'_exact_set_accuracy'])
            assert np.isclose(correct.all(axis=1)[nonempty].mean(),result[prefix+name+'_nonempty_exact_set_accuracy'])
    group=(recipe.get('loss_mode','unweighted'),records,seed)
    canonical={k:v for k,v in recipe.items() if k not in ['family','d','out']}
    if group in groups: assert canonical==groups[group]
    else: groups[group]=canonical
    checked.append(path.parent.name)
    diagnostics.append(diagnostic)
expected={f'w64-r4-t64-{family}-d{d}-s123{suffix}' for family in ['mamba2','gdn_current']
          for d in [1,3] for suffix in ['', '-balanced']}
assert set(checked)==expected
(ROOT/'object_binding_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
output=dict(passed=True,completed_runs=len(checked),runs=checked,
            checks=['independent row/predicate/label decoding','query-combination holdout disjointness',
                    'paired seen/heldout database and candidate order','regenerated validation/test hashes',
                    'saved raw accuracy, balanced accuracy and exact sets','matched recipes within condition/seed'])
(ROOT/'audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(f'Audited {len(checked)} completed runs: passed')

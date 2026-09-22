"""Independent token decoding and saved-prediction audit for the planned grid."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from boolean_recall import BooleanRecall

def digest(data):return hashlib.sha256(b''.join(a.tobytes() for a in data)).hexdigest()

expected=set(json.loads((ROOT/'expected_grid.json').read_text()))
paths=sorted(ROOT.glob('*/result.json'))
assert {p.parent.name for p in paths}==expected
cache,recipes,checked={}, {}, []
for path in paths:
    q=json.loads((path.parent/'recipe.json').read_text());r=json.loads(path.read_text())
    assert q['steps']==r['steps']==500 and r['complete']
    assert q['batch']==32 and q['width']==64 and q['test_examples']==4096
    key=(q['condition'],q['load'],q['seed'],q['sequence_length'])
    if key not in cache:
        task=BooleanRecall(q['condition'],q['sequence_length'])
        data=task.generate('boolean',q['load'],4096,np.random.default_rng(np.random.SeedSequence([q['seed'],30001,q['load']])))
        val=task.generate('boolean',q['load'],1024,np.random.default_rng(np.random.SeedSequence([q['seed'],10001,q['load']])))
        pairs=[]
        for tokens,positions,targets in zip(*data):
            starts=np.flatnonzero((tokens[:-5]>=4)&(tokens[:-5]<20))
            table={int(tokens[j]):int(tokens[j+1])-20 for j in starts}
            assert len(table)==q['load'] and set(table.values())<=set([0,1])
            assert tokens[-5]==1 and list(tokens[-2:])==[2,3]
            if q['condition']=='direct_and':pair=tokens[-4:-2]-20
            else:
                assert tokens[-4]!=tokens[-3]
                pair=np.array([table[int(tokens[-4])],table[int(tokens[-3])]])
            assert set(pair)<=set([0,1]);pairs.append(pair)
            if q['condition']=='retrieval':
                assert list(positions)==[len(tokens)-2,len(tokens)-1]
                assert np.array_equal(targets,pair+20)
            else:
                assert list(positions)==[len(tokens)-1]
                assert targets[0]==20+int(pair[0]==1 and pair[1]==1)
        pairs=np.asarray(pairs);case=2*pairs[:,0]+pairs[:,1]
        assert np.array_equal(np.bincount(case,minlength=4),[1024]*4)
        cache[key]=(data,val,pairs)
    data,val,pairs=cache[key]
    assert digest(data)==r['test_sha256'] and digest(val)==r['validation_sha256']
    a=np.load(path.parent/'test_predictions.npz')
    assert np.array_equal(a['targets'],data[2]) and np.array_equal(a['positions'],data[1])
    pred=a['predictions'];assert pred.shape==data[2].shape
    correct=pred==data[2]
    assert np.isclose(correct.mean(),r['accuracy'])
    assert np.isclose(correct.all(axis=1).mean(),r['exact_accuracy'])
    case=2*pairs[:,0]+pairs[:,1]
    for c in range(4):
        assert np.isclose(correct[case==c].mean(),r[f'truth_{c:02b}_accuracy'])
        assert np.isclose(correct[case==c].all(axis=1).mean(),r[f'truth_{c:02b}_exact_accuracy'])
    assert np.isclose(min(correct[case==c].all(axis=1).mean() for c in range(4)),r['worst_truth_exact_accuracy'])
    ones=(data[0][:,:-5]==21).sum(axis=1)
    n=q['load']
    if q['condition']=='retrieval':
        valid=np.isin(pred,[20,21]).all(axis=1)
        composed=(pred==21).all(axis=1)
        assert np.isclose((valid&(composed==pairs.all(axis=1))).mean(),r['composed_and_accuracy'])
        remaining_ones=ones[:,None]-pairs
        other=np.where(pred==21,remaining_ones/(n-1),np.where(pred==20,(n-1-remaining_ones)/(n-1),0))
        blind=20+(ones/n>.5).astype(int)
    else:
        positive=data[2]==21
        assert np.isclose((correct[positive].mean()+correct[~positive].mean())/2,r['balanced_accuracy'])
        total=n*(n-1)/2
        positive_pairs=ones*(ones-1)/2
        target_positive=pairs.all(axis=1)
        other=np.where(pred[:,0]==21,(positive_pairs-target_positive)/(total-1),
              np.where(pred[:,0]==20,(total-positive_pairs-~target_positive)/(total-1),0))
        blind=20+(positive_pairs/total>.5).astype(int)
    assert np.isclose(100*(correct.mean()-other.mean()),r['binding_gap_pp'])
    assert np.isclose((blind[:,None]==data[2]).mean(),r['query_blind_count_oracle_accuracy'])
    canonical={k:v for k,v in q.items() if k not in ['family','d','out']}
    if key in recipes:assert canonical==recipes[key]
    else:recipes[key]=canonical
    checked.append(path.parent.name)
for _,load,seed,length in cache:
    keys=[(c,load,seed,length) for c in ['retrieval','direct_and','retrieval_and']]
    if not all(k in cache for k in keys):continue
    retrieval,direct,both=[cache[k] for k in keys]
    assert np.array_equal(retrieval[0][0],both[0][0])
    assert np.array_equal(retrieval[0][0][:,:-5],direct[0][0][:,:-5])
    assert np.array_equal(retrieval[2],direct[2]) and np.array_equal(retrieval[2],both[2])
    assert np.array_equal((retrieval[0][2]==21).all(axis=1).astype(int)+20,both[0][2][:,0])
output=dict(passed=True,completed_runs=len(checked),runs=checked,
            checks=['complete planned grid','independent token/target decoding','equal truth-table test counts',
                    'regenerated validation/test hashes','saved accuracy/exact/truth-case/composed-AND/binding/control metrics',
                    'paired memories and bit pairs across conditions','matched recipes within condition/seed'])
(ROOT/'audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(f'Audited {len(checked)} runs: passed')

"""Recompute saved test scores and context-blind lookup controls, without a GPU."""
import json
from pathlib import Path
import numpy as np
from joint_recall import ROOT, CELLS, VALUE_BASE, VALUES, length, sha_file


def controls(condition, c, k):
    stem=ROOT/'data'/f'c{c}-k{k}-test'
    x=np.load(f'{stem}-{condition}.npy')
    y=np.load(f'{stem}-targets.npy')
    b=len(x);ih=1+np.arange(c)*(1+2*k)
    qh=length(c,k)//2+1+np.arange(c)*(1+2*k)
    ip=(ih[:,None]+1+2*np.arange(k)).reshape(-1)
    qp=(qh[:,None]+1+2*np.arange(k)).reshape(-1)
    keys=x[:,ip];values=x[:,ip+1];qkeys=x[:,qp]
    # All matching information values are supplied to this analytical control.
    match=qkeys[:,:,None]==keys[:,None,:]
    counts=np.stack([(match & (values[:,None,:]==v)).sum(-1)
                     for v in range(VALUE_BASE,VALUE_BASE+VALUES)],axis=-1)
    majority=counts.argmax(-1)+VALUE_BASE
    last_index=(match*np.arange(1,c*k+1)).max(-1)-1
    last=values[np.arange(b)[:,None],last_index]
    ctx_match=x[:,qh,None]==x[:,None,ih]
    ci=ctx_match.argmax(-1)
    ctxvals=values.reshape(b,c,k)[np.arange(b)[:,None],ci]
    ctxcounts=np.stack([(ctxvals==v).sum(-1) for v in range(VALUE_BASE,VALUE_BASE+VALUES)],-1)
    context_majority=np.repeat(ctxcounts.argmax(-1)+VALUE_BASE,k,axis=1)
    collision=y!=last
    return dict(key_only_majority_accuracy=float((majority==y).mean()),
                last_value_for_key_accuracy=float((last==y).mean()),
                context_only_majority_accuracy=float((context_majority==y).mean()),
                uniform_value_guess_accuracy=1/VALUES,
                overwrite_conflict_queries=int(collision.sum()),queries=int(y.size)),y,collision


def main():
    cached={(condition,f'c{c}-k{k}'):controls(condition,c,k)
            for condition in ['shared','unique'] for c,k,_ in CELLS}
    control_rows=[dict(condition=condition,cell=cell,**m)
                  for (condition,cell),(m,_,_) in cached.items()]
    (ROOT/'lookup_controls.json').write_text(json.dumps(control_rows,indent=2)+'\n')
    audited=[]
    for name in json.loads((ROOT/'planned_runs.json').read_text()):
        folder=ROOT/name
        if not (folder/'result.json').exists():continue
        r=json.loads((folder/'result.json').read_text())
        q=json.loads((folder/'recipe.json').read_text())
        assert r['complete'] and r['epochs']==32 and r['steps']==22624
        assert r['examples_seen']==5760000
        assert q['dataset_manifest_sha256']==sha_file(ROOT/'data/manifest.json')
        history=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [h['epoch'] for h in history]==list(range(1,33))
        assert [h['steps'] for h in history]==[707*i for i in range(1,33)]
        best=max(history,key=lambda h:h['accuracy'])
        assert best['epoch']==r['best_validation_epoch']
        assert best['accuracy']==r['best_validation_accuracy']
        for endpoint,subdir in [('final_test',folder),('best_test',folder/'best_test')]:
            acc=[];exact=[]
            for c,k,_ in CELLS:
                cell=f'c{c}-k{k}';data=np.load(subdir/f'test-{cell}.npz')
                _,y,collision=cached[q['condition'],cell]
                assert np.array_equal(data['targets'],y)
                assert np.array_equal(data['positions'],np.load(ROOT/'data'/f'{cell}-test-positions.npy'))
                good=data['predictions']==y
                a=float(good.mean());e=float(good.all(-1).mean());acc.append(a);exact.append(e)
                assert abs(a-r[endpoint]['cells'][cell]['accuracy'])<1e-12
                assert abs(e-r[endpoint]['cells'][cell]['exact_accuracy'])<1e-12
                audited.append(dict(run=name,endpoint=endpoint,cell=cell,accuracy=a,exact_accuracy=e,
                    overwrite_conflict_accuracy=float(good[collision].mean()) if collision.any() else None,
                    overwrite_conflict_queries=int(collision.sum())))
            assert abs(np.mean(acc)-r[endpoint]['accuracy'])<1e-12
            assert abs(np.mean(exact)-r[endpoint]['exact_accuracy'])<1e-12
    out=dict(passed=True,completed_runs=len(audited)//10,planned_runs=len(json.loads((ROOT/'planned_runs.json').read_text())),checks=audited)
    (ROOT/'result_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f'Recomputed saved predictions for {out["completed_runs"]}/{out["planned_runs"]} runs; controls saved.')


if __name__=='__main__':main()

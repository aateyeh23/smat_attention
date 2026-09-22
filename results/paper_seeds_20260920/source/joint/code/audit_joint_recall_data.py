"""Vectorized independent decoding, split disjointness and paired-control audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from joint_recall import ROOT, CELLS, sha_file, length

folder=ROOT/'data';manifest=json.loads((folder/'manifest.json').read_text())
for name,meta in manifest['files'].items():assert sha_file(folder/name)==meta['sha256'],name
checks=[]
for c,k,ntrain in CELLS:
    name=f'c{c}-k{k}';L=length(c,k);half=L//2
    info_headers=1+np.arange(c)*(1+2*k)
    query_headers=half+1+np.arange(c)*(1+2*k)
    ip=(info_headers[:,None]+1+2*np.arange(k)).reshape(-1)
    qp=(query_headers[:,None]+1+2*np.arange(k)).reshape(-1)
    seen={'unique':set(),'shared':set()}
    for split,expected in [('train',ntrain),('validation',1000),('test',2000)]:
        arrays={tag:np.load(folder/f'{name}-{split}-{tag}.npy',mmap_mode='r') for tag in ['unique','shared','positions','targets']}
        assert arrays['unique'].shape==arrays['shared'].shape==(expected,L)
        assert np.array_equal(arrays['positions'],np.broadcast_to(qp,(expected,c*k)))
        assert arrays['targets'].shape==(expected,c*k)
        for start in range(0,expected,5000):
            part={tag:a[start:start+5000] for tag,a in arrays.items()};b=len(part['targets'])
            a,z=part['unique'].copy(),part['shared'].copy()
            a[:,np.r_[ip,qp]]=0;z[:,np.r_[ip,qp]]=0
            assert np.array_equal(a,z)
            for condition in ['unique','shared']:
                x=part[condition]
                assert (x[:,0]==1).all() and (x[:,half]==2).all()
                assert (x[:,qp+1]==0).all()  # inquiry answer slots are blank
                ctx=x[:,info_headers];qctx=x[:,query_headers]
                assert ((ctx>=3)&(ctx<35)).all()
                assert (np.diff(np.sort(ctx,axis=1),axis=1)>0).all()
                assert np.array_equal(np.sort(ctx,axis=1),np.sort(qctx,axis=1))
                keys=x[:,ip].reshape(b,c,k);values=x[:,ip+1].reshape(b,c,k)
                qkeys=x[:,qp].reshape(b,c,k)
                assert ((keys>=35)&(keys<163)).all() and ((values>=163)&(values<179)).all()
                assert (np.diff(np.sort(keys,axis=2),axis=2)>0).all()
                if condition=='unique':assert (np.diff(np.sort(keys.reshape(b,-1),axis=1),axis=1)>0).all()
                else:assert (np.sort(keys,axis=2)==np.sort(keys,axis=2)[:,:1]).all()
                ctx_match=qctx[:,:,None]==ctx[:,None,:]
                assert (ctx_match.sum(axis=2)==1).all()
                ci=ctx_match.argmax(axis=2)
                tablekeys=keys[np.arange(b)[:,None],ci]
                tablevals=values[np.arange(b)[:,None],ci]
                key_match=qkeys[:,:,:,None]==tablekeys[:,:,None,:]
                assert (key_match.sum(axis=3)==1).all()
                ki=key_match.argmax(axis=3)
                decoded=np.take_along_axis(tablevals,ki,axis=2).reshape(b,-1)
                assert np.array_equal(decoded,part['targets'])
                assert np.array_equal(np.sort(qkeys,axis=2),np.sort(tablekeys,axis=2))
                for row in x:
                    h=hashlib.sha256(row.tobytes()).digest()
                    assert h not in seen[condition],(name,split,condition,'duplicate input')
                    seen[condition].add(h)
        checks.append(dict(cell=name,split=split,examples=expected,paired=True,decoded=True,disjoint=True))
assert sum(n for _,_,n in CELLS)==180000
output=dict(passed=True,train_examples_per_condition=180000,validation_examples_per_condition=5000,
            test_examples_per_condition=10000,steps_per_epoch=707,epochs=32,total_steps=22624,
            manifest_sha256=sha_file(folder/'manifest.json'),checks=checks)
(ROOT/'data_audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))

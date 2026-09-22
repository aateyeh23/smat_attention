"""Independent decoding and split/pairing audit for explicit-context datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
root=args.root;folder=root/'data';cfg=json.loads((root/'dataset_config.json').read_text())
manifest=json.loads((folder/'manifest.json').read_text());assert cfg['layout']=='record'
vocab=manifest['vocab'];value_base=vocab-16;key_base=35
assert value_base==key_base+cfg.get('key_count',128)
assert sum(row[2] for row in cfg['cells'])==180000
for name,meta in manifest['files'].items():
    assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==meta['sha256'],name
checks=[]
for c,k,ntrain in cfg['cells']:
    records=c*k;seen={'shared':set(),'unique':set()}
    for split,n in [('train',ntrain),('validation',cfg['validation_per_cell']),('test',cfg['test_per_cell'])]:
        stem=f'c{c}-k{k}-{split}'
        a={t:np.load(folder/f'{stem}-{t}.npy',mmap_mode='r') for t in ['unique','shared','positions','targets']}
        L=a['unique'].shape[1];half=L//2
        assert a['shared'].shape==a['unique'].shape==(n,L)
        assert np.array_equal(a['positions'],np.broadcast_to(half+2+3*np.arange(records),(n,records)))
        assert a['targets'].shape==(n,records)
        for start in range(0,n,1000):
            part={t:v[start:start+1000] for t,v in a.items()};b=len(part['targets'])
            for condition in ['shared','unique']:
                x=part[condition]
                info=x[:,1:1+3*records].reshape(b,records,3)
                query=x[:,half+1:half+1+3*records].reshape(b,records,3)
                assert (x[:,0]==1).all() and (x[:,half]==2).all()
                assert not x[:,1+3*records:half].any() and not x[:,half+1+3*records:].any()
                assert not query[:,:,2].any()
                assert ((info[:,:,0]>=3)&(info[:,:,0]<35)).all()
                assert ((info[:,:,1]>=key_base)&(info[:,:,1]<value_base)).all()
                assert ((info[:,:,2]>=value_base)&(info[:,:,2]<vocab)).all()
                contexts=np.sort(info[:,:,0],axis=1)
                assert np.array_equal(contexts,np.repeat(contexts[:,::k],k,axis=1))
                assert (np.diff(contexts[:,::k],axis=1)>0).all()
                lookup=info[:,:,0].astype('int64')*vocab+info[:,:,1]
                qkey=query[:,:,0].astype('int64')*vocab+query[:,:,1]
                order=lookup.argsort(1);sorted_keys=np.take_along_axis(lookup,order,1);qo=qkey.argsort(1)
                assert np.array_equal(sorted_keys,np.take_along_axis(qkey,qo,1))
                assert (np.diff(sorted_keys,axis=1)>0).all()
                decoded=np.take_along_axis(info[:,:,2],order,1)
                assert np.array_equal(decoded,np.take_along_axis(part['targets'],qo,1))
                keys=np.sort(info[:,:,1],axis=1)
                if condition=='unique':assert (np.diff(keys,axis=1)>0).all()
                else:
                    assert np.array_equal(keys,np.repeat(keys[:,::c],c,axis=1))
                    assert (np.diff(keys[:,::c],axis=1)>0).all()
                for row in x:
                    h=hashlib.sha256(row.tobytes()).digest()
                    assert h not in seen[condition];seen[condition].add(h)
            u=part['unique'].copy();s=part['shared'].copy()
            u[:,2:1+3*records:3]=s[:,2:1+3*records:3]=0
            u[:,half+2:half+1+3*records:3]=s[:,half+2:half+1+3*records:3]=0
            assert np.array_equal(u,s)
        checks.append(dict(cell=f'c{c}-k{k}',split=split,examples=n,decoded=True,paired=True,disjoint=True))
(root/'data_audit.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
print(str(root)+': all targets, paired controls, hashes and split checks passed')

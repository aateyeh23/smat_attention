"""Degree-preserving rewiring conditioned on exact single-summary VC dimension 2.

This is a bounded constrained random walk initialized from a relabeled affine
plane, not a uniform sampler over all degree-matched VC-2 set systems.
"""
import dataclasses
import hashlib
import itertools
import json
import sys
from pathlib import Path
import numpy as np
from common import ROOT, SOURCE, atomic_json


def has_shattered_triple(matrix, return_witness=False):
    """Exhaustive triple test, pruned by necessary pair multiplicity >=2.

    A shattered triple must be contained in a row (pattern 111), and each of
    its three pairs must also occur in another row (patterns 110,101,011).
    Every surviving candidate is tested for all eight patterns directly.
    """
    flat = matrix.reshape(-1, matrix.shape[-1]).astype(np.int16)
    degree = int(flat[0].sum())
    assert np.all(flat.sum(1) == degree)
    members = np.flatnonzero(flat).reshape(flat.shape[0], degree) % flat.shape[1]
    offsets = np.asarray(list(itertools.combinations(range(degree), 3)))
    triples = members[:, offsets].reshape(-1, 3)
    gram = flat.T@flat
    keep = ((gram[triples[:,0],triples[:,1]] >= 2)
            & (gram[triples[:,0],triples[:,2]] >= 2)
            & (gram[triples[:,1],triples[:,2]] >= 2))
    triples = triples[keep]
    for start in range(0,len(triples),1024):
        candidates=triples[start:start+1024]
        codes=(flat[:,candidates]*np.array([1,2,4])).sum(-1)
        patterns=np.bitwise_or.reduce(np.left_shift(np.uint16(1),codes),axis=0)
        hit=np.flatnonzero(patterns==255)
        if len(hit):
            return candidates[hit[0]].tolist() if return_witness else True
    return None if return_witness else False


def geometric(q):
    points=np.stack([np.arange(q*q)//q,np.arange(q*q)%q],1)
    nonzero=points!=0
    lead=nonzero.argmax(1)
    directions=points[nonzero.any(1)&(points[np.arange(q*q),lead]==1)]
    offsets=(directions@points.T)%q
    return np.eye(q,dtype=np.uint8)[offsets].transpose(0,2,1).copy()


def make_matrix(q, topology, layer, length, proposals=500):
    rng=np.random.default_rng(np.random.SeedSequence([topology,layer,length,2202]))
    original=geometric(q)
    permutation=rng.permutation(q*q)
    matrix=original[:,:,permutation].copy()
    # Mix pairs of groups within one parallel class. These are genuine rewires,
    # and not a common permutation of column labels.
    direction=int(rng.integers(q+1))
    order=rng.permutation(q)
    for i in range(0,q-1,2):
        a,b=map(int,order[i:i+2])
        pool=np.flatnonzero(matrix[direction,a]|matrix[direction,b])
        pool=rng.permutation(pool)
        matrix[direction,a]=0;matrix[direction,b]=0
        matrix[direction,a,pool[:q]]=1;matrix[direction,b,pool[q:]]=1
    assert not has_shattered_triple(matrix)
    accepted=0
    for _ in range(proposals):
        e=int(rng.integers(q+1));a,b=map(int,rng.choice(q,2,replace=False))
        x=int(rng.choice(np.flatnonzero(matrix[e,a])))
        y=int(rng.choice(np.flatnonzero(matrix[e,b])))
        matrix[e,a,x]=matrix[e,b,y]=0
        matrix[e,a,y]=matrix[e,b,x]=1
        flat=matrix.reshape(-1,q*q)
        unique=len(np.unique(flat,axis=0))==q*(q+1)
        if unique and not has_shattered_triple(matrix):
            accepted+=1
        else:
            matrix[e,a,x]=matrix[e,b,y]=1
            matrix[e,a,y]=matrix[e,b,x]=0
    sys.path.insert(0,str(SOURCE/'smat'))
    from vc_dimension import pseudo_dimension_detailed
    result=pseudo_dimension_detailed(matrix.reshape(-1,q*q),time_limit=30.)
    assert result.exact and result.dimension==2 and result.witness.verify(matrix.reshape(-1,q*q))
    assert not has_shattered_triple(matrix)
    assert np.all(matrix.sum(-1)==q)
    assert np.all(matrix.sum(1)==1)
    assert np.all(matrix.sum((0,1))==q+1)
    assert len(np.unique(matrix.reshape(-1,q*q),axis=0))==q*(q+1)
    gram=matrix.reshape(-1,q*q).astype(np.int16).T@matrix.reshape(-1,q*q).astype(np.int16)
    overlaps=gram[np.triu_indices(q*q,1)]
    assert np.any(overlaps!=1), 'This would only be an affine plane under relabeling'
    values,counts=np.unique(overlaps,return_counts=True)
    output=matrix.astype(np.float32)
    return output,dict(q=q,topology_seed=topology,layer=layer,length=length,
        proposals=proposals,accepted_swaps=accepted,initial_mixed_direction=direction,
        global_column_permutation=permutation.tolist(),vc_dimension=2,vc_exact=True,
        independent_exhaustive_triple_check=True,witness=dataclasses.asdict(result.witness),
        unique_summary_count=q*(q+1),profile_count=q*q,row_degree=q,column_degree=q+1,
        incidence_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
        changed_memberships_vs_original=int(np.count_nonzero(output!=original)),
        changed_memberships_vs_relabeled_geometry=int(np.count_nonzero(output!=original[:,:,permutation])),
        pairwise_profile_overlap_histogram={str(int(v)):int(n) for v,n in zip(values,counts)})


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    base=ROOT.parent/'joint_incidence_20260921/random-t17/shared-gdn_current-d3-w64-s123/ablation.json'
    geometry=json.loads(base.read_text())['matrix_tables']
    arrays={};records=[]
    for row in geometry:
        matrix,audit=make_matrix(row['q'],17,row['layer'],row['length'])
        key=f"l{row['layer']}_t{row['length']}"
        arrays[key]=matrix;records.append(audit)
        print(json.dumps({k:v for k,v in audit.items() if k not in ('witness','global_column_permutation')}),flush=True)
    np.savez_compressed(ROOT/'incidence.npz',**arrays)
    atomic_json(ROOT/'construction.json',dict(method='Randomly relabeled affine plane; pairwise group mixing in one random direction; 500 degree-preserving swap proposals accepting only unique-summary VC<=2 states. Exact final VC=2 certified independently.',
        uniform_over_vc2_families=False,scope='Individual summary supports over profiles; not the learned four-read union family',matrices=records))


if __name__=='__main__':
    main()

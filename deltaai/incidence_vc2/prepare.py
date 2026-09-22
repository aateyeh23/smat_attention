"""Fix three seeds and provenance without modifying the ongoing first campaign."""
import hashlib
import json
from pathlib import Path
import shutil
from common import BASE,CODE,DATA,ROOT,SEEDS,atomic_json


def main():
    assert (ROOT/'construction.json').exists()
    assert not (ROOT/'source_sha256.json').exists(), 'Already frozen'
    tasks=[]
    root=ROOT/'random_vc2-t17';root.mkdir(exist_ok=True)
    (root/'data').symlink_to(DATA/'data',target_is_directory=True)
    shutil.copy2(DATA/'dataset_config.json',root/'dataset_config.json')
    for index,seed in enumerate(SEEDS):
        folder=root/f'shared-gdn_current-d3-w64-s{seed}'
        tasks.append(dict(name=f'random_vc2-t17-s{seed}',variant='random_vc2',topology=17,
            seed=seed,root=str(root),result=str(folder/'result.json'),
            script=str(CODE/'run.py'),args=['--index',str(index)]))
    atomic_json(ROOT/'tasks.json',tasks)
    protocol=dict(parent_campaign=str(BASE),training_seeds=SEEDS,topology_seed=17,
        variant='random_vc2',scope='Exact VC dimension 2 for individual summary supports over profiles; the four-summary union family is not constrained to VC 2.',
        counts='Same profile count, summary count, distinct summary count, row degree and column degree as geometry and original random arms.',
        construction='Random column relabeling of geometric incidence; mix pairs of groups in one random partition; 500 degree-preserving swap proposals accepted iff no duplicate summary and no shattered triple. Exact final VC=2 required.',
        caveat='Constrained random walk from geometry, not uniform sampling over all VC-2 families. Column relabeling also changes alignment to the learned coordinate hash.',
        model=dict(family='gdn_current',d=3,width=64,layers=2,head_dim=16,state_dim=16,writes=1,reads=4),
        training=dict(epochs=32,batch=256,lr=.003,weight_decay=.1,steps=22560,examples_per_epoch=180000,early_stopping=False),
        primary='Final-epoch test accuracy at 128 bindings; pair by training seed against original geometry and random-t17.',
        secondary='All five loads, macro accuracy, exact-table accuracy; no test-based selection of matrices or checkpoints.',
        data_manifest_sha256=hashlib.sha256((DATA/'data/manifest.json').read_bytes()).hexdigest())
    atomic_json(ROOT/'protocol.json',protocol)
    sources=json.loads((BASE/'source_sha256.json').read_text())
    for p,h in sources.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    extra=[*CODE.glob('*.py'),*CODE.glob('*.sbatch'),ROOT/'incidence.npz',ROOT/'construction.json',ROOT/'tasks.json',ROOT/'protocol.json']
    for p in extra:sources[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    atomic_json(ROOT/'source_sha256.json',sources)
    (ROOT/'logs').mkdir(exist_ok=True)
    print('Prepared three fresh full-budget VC-2 runs',flush=True)


if __name__=='__main__':main()

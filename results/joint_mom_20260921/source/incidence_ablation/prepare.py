"""Freeze sources and fix the 15-run comparison before any training."""
import hashlib
import json
from pathlib import Path
import shutil
from common import CODE, ROOT, SOURCE, DATA, SEEDS, TOPOLOGIES, atomic_json


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'source_sha256.json').exists(), 'Experiment already frozen'
    origin = CODE.parent/'results/joint_recall_iterations/source_v5'
    for name, expected in json.loads((origin/'manifest.json').read_text()).items():
        assert hashlib.sha256((origin/name).read_bytes()).hexdigest() == expected, name
    shutil.copytree(origin, SOURCE, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(CODE.parent/'joint_recall_campaign.py', SOURCE/'joint_recall_campaign.py')
    shutil.copytree('~/zoology/zoology', SOURCE/'zoology',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    tasks = []
    for seed in SEEDS:
        for variant, topology in [('geometry', 0), ('random', TOPOLOGIES[0]),
                                  ('buckets_profiles', 0), ('buckets_bytes', 0),
                                  ('random', TOPOLOGIES[1])]:
            name = variant+(f'-t{topology}' if topology else '')
            root = ROOT/name
            root.mkdir(exist_ok=True)
            (root/'data').symlink_to(DATA/'data', target_is_directory=True) if not (root/'data').exists() else None
            shutil.copy2(DATA/'dataset_config.json', root/'dataset_config.json')
            folder = root/f'shared-gdn_current-d3-w64-s{seed}'
            tasks.append(dict(name=f'{name}-s{seed}', variant=variant, topology=topology,
                seed=seed, root=str(root), result=str(folder/'result.json'),
                script=str(CODE/'run.py'), args=['--index', str(len(tasks))]))
    atomic_json(ROOT/'tasks.json', tasks)
    (ROOT/'logs').mkdir(exist_ok=True)
    files = sorted([*SOURCE.rglob('*.py'), *CODE.glob('*.py'), *CODE.glob('*.sbatch')])
    atomic_json(ROOT/'source_sha256.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    atomic_json(ROOT/'protocol.json', dict(seeds=SEEDS, topology_seeds=TOPOLOGIES,
        task_count=len(tasks), primary='geometry minus random, average topology draws within training seed',
        dataset=str(DATA), dataset_manifest_sha256=hashlib.sha256((DATA/'data/manifest.json').read_bytes()).hexdigest(),
        training=dict(width=64,layers=2,head_dim=16,state_dim=16,epochs=32,batch=256,
                      lr=.003,weight_decay=.1,examples_per_epoch=180000,total_steps=22560),
        primary_metric='Final-epoch per-query test accuracy at 128 bindings',
        secondary_metrics='All five loads, equal-cell macro, exact-table accuracy; no selection by test',
        no_early_stopping=True,
        native_reference='Existing frozen native GDN results only; not a new memory-matched control',
        bucket_controls=dict(buckets_profiles='Same q^2 profiles, unchanged writer, direct 4-bucket reader; fewer reader parameters and lower physical table bytes',
                             buckets_bytes='q(2q+1) independent states; matches profile+summary FP32 matrix-table bytes, changes hash bin counts and reader size'),
        limitations=['Only geometry/random isolate incidence with identical parameter and state shapes.',
                     'Matrix-table bytes are not measured autoregressive cache bytes.',
                     '15 independent fresh training runs; two graph draws crossed with three training seeds.',
                     'Existing data distribution was used in development; results are not OOD generalization.']))
    print(json.dumps(dict(root=str(ROOT),runs=len(tasks))))


if __name__ == '__main__':
    main()

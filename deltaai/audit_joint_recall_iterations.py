"""Recompute every finished development score from saved predictions."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'results/joint_recall_iterations'
tasks=[t for t in json.loads((ROOT/'campaign.json').read_text()) if t.get('kind','training')=='training'];checked=[]
for task in tasks:
    path=Path(task['result'])
    if not path.exists():continue
    folder=path.parent;root=folder.parent
    result=json.loads(path.read_text());recipe=json.loads((folder/'recipe.json').read_text())
    manifest_path=root/'data/manifest.json';manifest=json.loads(manifest_path.read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==recipe['dataset_manifest_sha256']
    train_count=sum(cell['train_examples'] for cell in manifest['cells'])
    per_epoch=sum(math.ceil(cell['train_examples']/recipe['batch']) for cell in manifest['cells'])
    assert result['complete'] and result['epochs']==recipe['epochs']==32
    assert result['steps']==recipe['total_steps']==per_epoch*32
    assert result['examples_seen']==recipe['example_presentations']==train_count*32==5760000
    history=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    assert [h['epoch'] for h in history]==list(range(1,33))
    assert [h['steps'] for h in history]==[per_epoch*i for i in range(1,33)]
    best=max(history,key=lambda h:h['accuracy'])
    assert best['epoch']==result['best_validation_epoch']
    assert best['accuracy']==result['best_validation_accuracy']
    for endpoint,source in [('final_test',folder),('best_test',folder/'best_test')]:
        accuracy=[];exact=[]
        for cell in manifest['cells']:
            name=cell['name'];data=np.load(source/f'test-{name}.npz')
            y=np.load(root/'data'/f'{name}-test-targets.npy',mmap_mode='r')
            positions=np.load(root/'data'/f'{name}-test-positions.npy',mmap_mode='r')
            assert np.array_equal(y,data['targets']) and np.array_equal(positions,data['positions'])
            assert data['predictions'].shape==y.shape
            good=data['predictions']==y;a=float(good.mean());e=float(good.all(-1).mean())
            assert abs(a-result[endpoint]['cells'][name]['accuracy'])<1e-12
            assert abs(e-result[endpoint]['cells'][name]['exact_accuracy'])<1e-12
            accuracy.append(a);exact.append(e)
            checked.append(dict(task=task['name'],endpoint=endpoint,cell=name,accuracy=a,exact_accuracy=e,
                                epochs=32,steps=result['steps'],examples_seen=result['examples_seen']))
        assert abs(np.mean(accuracy)-result[endpoint]['accuracy'])<1e-12
        assert abs(np.mean(exact)-result[endpoint]['exact_accuracy'])<1e-12
        assert math.isfinite(result[endpoint]['accuracy'])
out=dict(passed=True,completed=len({r['task'] for r in checked}),scheduled=len(tasks),scores=checked)
(ROOT/'result_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(f'{out["completed"]}/{out["scheduled"]} completed runs: budgets, checkpoint selection and saved predictions verified')

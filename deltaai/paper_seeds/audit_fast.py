"""Audit fast-backend provenance, unchanged recipes, full budgets and reports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent.parent/'results/paper_fast_20260920'
dense=root.with_name('paper_seeds_20260920')
def read(path):return json.loads(path.read_text())
def finite(value):
    if isinstance(value,float):assert math.isfinite(value)
    elif isinstance(value,dict):
        for x in value.values():finite(x)
    elif isinstance(value,list):
        for x in value:finite(x)

parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true')
args=parser.parse_args()
tasks=read(root/'tasks.json')
assert len(tasks)==10 and {(t['family'],t['seed']) for t in tasks}=={(f,s) for f in ['mamba2','gdn'] for s in [123,1,2,3,4]}
manifest=root/'source_sha256.json'
for relative,expected in read(manifest).items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative
for relative,expected in read(dense/'source_sha256.json').items():
    assert hashlib.sha256((dense/relative).read_bytes()).hexdigest()==expected,relative
complete=[];missing=[];values={}
for task in tasks:
    path=Path(task['result']);folder=path.parent
    if not path.exists():missing.append(task['name']);continue
    result=read(path);finite(result)
    assert result['complete'] and result['epochs']==32 and result['steps']==22560 and result['examples_seen']==5760000
    recipe=read(folder/'recipe.json')
    assert recipe==read(dense/'joint_loglinear'/folder.name/'recipe.json'),folder
    extra=read(folder/'fast_backend_recipe.json')
    assert extra['seed']==task['seed'] and extra['family']==task['family'] and extra['backend']==task['backend']
    assert extra['initialization']=='fresh from seed' and extra['levels']==13
    assert extra['source_manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result['parameters']==result['trainable_parameters']==(105984 if task['family']=='mamba2' else 61468)
    history=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    finite(history)
    assert [r['epoch'] for r in history]==list(range(1,33))
    assert all(r['steps']==r['epoch']*705 for r in history)
    assert (folder/'checkpoint.pt').is_file()
    for metrics in [history[-1],result['final_test']]:
        assert set(metrics['cells'])=={'c1-k4','c2-k8','c8-k16','c16-k16','c32-k16'}
        assert abs(metrics['accuracy']-statistics.mean(c['accuracy'] for c in metrics['cells'].values()))<1e-12
    for cell in result['final_test']['cells']:assert (folder/f'test-{cell}.npz').is_file()
    values[(task['family'],task['seed'])]=(100*history[-1]['accuracy'],100*result['final_test']['accuracy'])
    complete.append(task['name'])
with (root/'per_seed.csv').open() as stream:rows=list(csv.DictReader(stream))
assert len(rows)==len(complete)
for row in rows:
    expected=values[(row['family'],int(row['seed']))]
    assert float(row['validation_accuracy'])==expected[0] and float(row['test_accuracy'])==expected[1]
for row in read(root/'summary.json'):
    group=[v[0] for (family,seed),v in values.items() if family==row['family']]
    assert row['n']==len(group) and row['mean']==statistics.mean(group)
    if len(group)>1:assert row['std']==statistics.stdev(group)
report=dict(requested_runs=10,completed_runs=len(complete),completed_checks_passed=True,complete=not missing,missing=missing)
(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if args.require_complete and missing:raise SystemExit(10)

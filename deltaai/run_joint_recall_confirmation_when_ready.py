"""Lock the planned final checkpoints, then evaluate on fresh shared tables.

This queue task runs after every training task has been dispatched. It can wait
on the remaining trainers using an otherwise idle GPU, and leaves the task
unfinished for the next allocation if insufficient time remains for evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


p=argparse.ArgumentParser();p.add_argument('--max-minutes',type=float,default=110);args=p.parse_args()
root=Path(__file__).resolve().parent/'results/joint_recall_iterations'
plan_path=root/'confirmation_plan.json';selection=json.loads(plan_path.read_text())
models=selection['models']+selection.get('additional_models',[])
assert len(models)==9 and {(m['family'],m['d']) for m in selection['models']}=={
    (family,d) for family in ['mamba2','gdn_current'] for d in [1,2,3,4]}
deadline=time.monotonic()+args.max_minutes*60
assert json.loads((root/'all_d_gate.json').read_text())['passed']
print('WAITING_FOR_SELECTED_CHECKPOINTS',flush=True)
while not all((Path(m['folder'])/'result.json').exists() for m in models):
    if deadline-time.monotonic()<15*60:
        print('CONFIRMATION_DEFERRED: fewer than15 minutes remain; no fresh data generated.',flush=True)
        raise SystemExit(0)
    time.sleep(15)
if deadline-time.monotonic()<15*60:
    print('CONFIRMATION_DEFERRED: fewer than15 minutes remain; no fresh data generated.',flush=True)
    raise SystemExit(0)

manifest=Path(selection['dataset_root'])/'data/manifest.json'
assert digest(manifest)==selection['dataset_manifest_sha256']
history={}
for chosen in models:
    folder=Path(chosen['folder']);recipe=json.loads((folder/'recipe.json').read_text())
    result=json.loads((folder/'result.json').read_text())
    assert recipe['family']==chosen['family'] and recipe['d']==chosen['d']
    assert recipe['condition']=='shared' and recipe['seed']==123
    assert recipe['layers']==2 and recipe['width']==64 and recipe['batch']==256
    assert recipe['epochs']==result['epochs']==32 and result['complete']
    assert result['steps']==recipe['total_steps']==22560
    assert result['examples_seen']==recipe['example_presentations']==5760000
    assert recipe['dataset_manifest_sha256']==selection['dataset_manifest_sha256']
    assert recipe['lr']==(.01 if chosen.get('label')=='gdn_current-d1-lr01' else .003)
    assert recipe['weight_decay']==.1 and recipe.get('gdn_memory_scalar_decay',True)
    chosen['checkpoint_sha256']=digest(folder/'checkpoint.pt')
    chosen['recipe_sha256']=digest(folder/'recipe.json')
    chosen['training_result_sha256']=digest(folder/'result.json')
    rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in rows]==list(range(1,33))
    history[chosen.get('label',f'{chosen["family"]}-d{chosen["d"]}')]=rows[-1]
oracle=next(c['best_single_field'] for c in json.loads(
    (root/'explicit_context_capacity512/validation_oracle_summary.json').read_text())['cells'] if c['cell']=='c32-k16')
for family in ['mamba2','gdn_current']:
    smat=history[f'{family}-d3']
    native=[row for label,row in history.items() if label.startswith(f'{family}-d1')]
    assert smat['accuracy']>max(row['accuracy'] for row in native)
    assert smat['cells']['c32-k16']['accuracy']>max(oracle,max(row['cells']['c32-k16']['accuracy'] for row in native))

selection['selection_plan_sha256']=digest(plan_path)
selection['selection_script_sha256']=digest(Path(__file__))
output=root/'confirmation_capacity512_all_d';output.mkdir(exist_ok=True)
lock=output/'selection.json'
if lock.exists():assert json.loads(lock.read_text())==selection
else:
    temp=lock.with_suffix('.tmp');temp.write_text(json.dumps(selection,indent=2)+'\n');temp.replace(lock)
print('SELECTION_LOCKED',lock,flush=True)
cmd=[sys.executable,'-u',str(Path(__file__).resolve().parent/'confirm_joint_recall.py'),
     '--selection',str(lock),'--output',str(output),'--split-code','9001','--examples-per-cell','10000',
     '--max-minutes',str((deadline-time.monotonic())/60)]
raise SystemExit(subprocess.run(cmd).returncode)

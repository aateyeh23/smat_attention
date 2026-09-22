"""Evaluate a locked validation-selected comparison on fresh independent tables.

Run on an allocated GPU. Selection identifies native and SMat final checkpoints trained
on identical datasets, and the frozen source tree used to construct the models.
Bootstrap intervals concern test sampling conditional on the single training seed.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from joint_recall_controls import single_field_oracles

p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--split-code',type=int,default=9001)
p.add_argument('--examples-per-cell',type=int,default=10000)
p.add_argument('--max-minutes',type=float,default=110)  # Campaign runner compatibility.
args=p.parse_args();selection=json.loads(args.selection.read_text());args.output.mkdir(parents=True,exist_ok=True)
lock=args.output/'selection.json'
if lock.exists():assert json.loads(lock.read_text())==selection
else:lock.write_text(json.dumps(selection,indent=2)+'\n')
evaluation_source=args.output/'evaluation_source';evaluation_source.mkdir(exist_ok=True)
evaluation_hashes={}
for name in ['confirm_joint_recall.py','joint_recall_controls.py','audit_joint_recall_confirmation.py','collect_joint_recall_confirmation.py','run_joint_recall_confirmation_when_ready.py']:
    origin=Path(__file__).resolve().parent/name;destination=evaluation_source/name
    if destination.exists():assert destination.read_bytes()==origin.read_bytes(),name
    else:shutil.copyfile(origin,destination)
    evaluation_hashes[name]=hashlib.sha256(destination.read_bytes()).hexdigest()
(evaluation_source/'manifest.json').write_text(json.dumps(evaluation_hashes,indent=2)+'\n')
assert args.split_code not in [1001,2001,3001]
source=Path(selection['source']);sys.path.insert(0,str(source/'deltaai'))
sys.path.insert(0,str(source/'smat'))
source_manifest=json.loads((source/'manifest.json').read_text())
for name,digest in source_manifest.items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
import joint_recall as jr

dataset_root=Path(selection['dataset_root']);jr.configure_dataset(dataset_root)
manifest_hash=jr.sha_file(dataset_root/'data/manifest.json')
assert manifest_hash==selection['dataset_manifest_sha256']
ds=sorted({m['d'] for m in selection['models']})
assert 1 in ds and 3 in ds and set(ds).issubset({1,2,3,4})
expected={(family,d) for family in ['mamba2','gdn_current'] for d in ds}
assert len(selection['models'])==len(expected)
assert {(m['family'],m['d']) for m in selection['models']}==expected
models=selection['models']+selection.get('additional_models',[])
labels=[m.get('label',f'{m["family"]}-d{m["d"]}') for m in models]
assert len(set(labels))==len(labels)
assert all(m['d']==1 and m['family'] in ['mamba2','gdn_current'] and 'label' in m
           for m in selection.get('additional_models',[]))
for chosen in models:
    folder=Path(chosen['folder']);recipe=json.loads((folder/'recipe.json').read_text())
    result=json.loads((folder/'result.json').read_text())
    assert recipe['dataset_manifest_sha256']==manifest_hash
    assert recipe['condition']=='shared' and result['complete'] and result['epochs']==32
    assert recipe['example_presentations']==result['examples_seen']==5760000
    assert recipe['total_steps']==result['steps'] and recipe['seed']==123
    assert jr.sha_file(folder/'checkpoint.pt')==chosen['checkpoint_sha256']
    assert recipe['family']==chosen['family'] and recipe['d']==chosen['d']

data=[];controls={};context_controls={};single_field_controls={};files={}
for c,k,_ in jr.CELLS:
    _,x,positions,targets=jr.generate(c,k,args.examples_per_cell,args.split_code)
    name=jr.cell_name(c,k)
    for tag,a in [('inputs',x),('positions',positions),('targets',targets)]:
        path=args.output/f'fresh-{name}-{tag}.npy';np.save(path,a);files[path.name]=jr.sha_file(path)
    data.append((x,positions,targets))
    oracles=single_field_oracles(x,c,k,jr.LAYOUT,jr.VOCAB,jr.VALUES)
    controls[name]=oracles['key_only'];context_controls[name]=oracles['context_only']
    single_field_controls[name]=oracles['best_single_field']
    path=args.output/f'oracles-{name}.npz';np.savez(path,**oracles);files[path.name]=jr.sha_file(path)
jr.atomic_json(args.output/'data_manifest.json',dict(split_code=args.split_code,
    examples_per_cell=args.examples_per_cell,training_manifest_sha256=manifest_hash,
    evaluation_source_sha256=evaluation_hashes,files=files))

torch.set_num_threads(2);predictions={};metrics={}
for chosen in models:
    folder=Path(chosen['folder']);recipe=json.loads((folder/'recipe.json').read_text())
    label=chosen.get('label',f'{chosen["family"]}-d{chosen["d"]}');out=args.output/label;out.mkdir(exist_ok=True)
    ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==32 and ck['cursor']==0
    model=jr.make_model(recipe['family'],recipe['width'],recipe['d'],recipe['lengths'],jr.VOCAB)
    if hasattr(jr,'apply_model_options'):jr.apply_model_options(model,recipe)
    elif recipe.get('gdn_memory_scalar_decay') is False:raise ValueError('Selected source cannot reconstruct memory option')
    model.load_state_dict(ck['model'])
    for name,module in model.named_modules():
        if hasattr(module,'_steps'):module._steps=ck['module_steps'].get(name,0)
    metrics[label]=jr.evaluate_all(model,data,recipe['eval_batch'],out)
    for (c,k,_),(_,_,targets) in zip(jr.CELLS,data):
        cell=jr.cell_name(c,k);a=np.load(out/f'test-{cell}.npz')
        assert np.array_equal(a['targets'],targets)
        good=a['predictions']==targets
        assert abs(good.mean()-metrics[label]['cells'][cell]['accuracy'])<1e-12
        predictions[label,cell]=good.mean(-1)
    print('CONFIRM',label,json.dumps(metrics[label]),flush=True)
    del model,ck;gc.collect();torch.cuda.empty_cache()

def interval(delta,rng):
    samples=np.concatenate([delta[rng.integers(len(delta),size=(100,len(delta)))].mean(1) for _ in range(20)])
    return dict(difference=float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist()),samples

comparisons=[];macro_comparisons=[];rng=np.random.default_rng(20260921)
for family in ['mamba2','gdn_current']:
    for d in [value for value in ds if value!=1]:
        family_rows=[];native_bootstraps=[];control_bootstraps=[];single_field_bootstraps=[]
        for c,k,_ in jr.CELLS:
            cell=jr.cell_name(c,k);native=predictions[f'{family}-d1',cell];smat=predictions[f'{family}-d{d}',cell]
            native_interval,native_samples=interval(smat-native,rng)
            control_interval,control_samples=interval(smat-controls[cell],rng)
            single_interval,single_samples=interval(smat-single_field_controls[cell],rng)
            row=dict(family=family,d=d,cell=cell,native_accuracy=float(native.mean()),
                smat_accuracy=float(smat.mean()),key_only_majority_accuracy=float(controls[cell].mean()),
                context_only_majority_accuracy=float(context_controls[cell].mean()),
                best_single_field_accuracy=float(single_field_controls[cell].mean()),
                smat_minus_native=native_interval,smat_minus_key_only=control_interval,
                smat_minus_single_field=single_interval)
            comparisons.append(row);family_rows.append(row)
            native_bootstraps.append(native_samples);control_bootstraps.append(control_samples)
            single_field_bootstraps.append(single_samples)
        # Equal-cell macro accuracy, with independent paired table resampling within
        # each cell. Reuse the independent bootstrap draws already made above.
        macro=dict(family=family,d=d,cells=[r['cell'] for r in family_rows])
        for key in ['native_accuracy','smat_accuracy','key_only_majority_accuracy','context_only_majority_accuracy','best_single_field_accuracy']:
            macro[key]=float(np.mean([r[key] for r in family_rows]))
        for key,samples in [('smat_minus_native',native_bootstraps),('smat_minus_key_only',control_bootstraps),('smat_minus_single_field',single_field_bootstraps)]:
            macro[key]=dict(difference=float(np.mean([r[key]['difference'] for r in family_rows])),
                            ci95=np.quantile(np.mean(samples,axis=0),[.025,.975]).tolist())
        macro_comparisons.append(macro)
additional_comparisons=[];additional_macro_comparisons=[]
for chosen in selection.get('additional_models',[]):
    family=chosen['family'];label=chosen['label']
    for d in [value for value in ds if value!=1]:
        rows=[];draws=[]
        for c,k,_ in jr.CELLS:
            cell=jr.cell_name(c,k);native=predictions[label,cell];smat=predictions[f'{family}-d{d}',cell]
            estimate,samples=interval(smat-native,rng)
            row=dict(family=family,d=d,native_label=label,cell=cell,native_accuracy=float(native.mean()),
                     smat_accuracy=float(smat.mean()),smat_minus_native=estimate)
            rows.append(row);draws.append(samples);additional_comparisons.append(row)
        additional_macro_comparisons.append(dict(family=family,d=d,native_label=label,
            native_accuracy=float(np.mean([row['native_accuracy'] for row in rows])),
            smat_accuracy=float(np.mean([row['smat_accuracy'] for row in rows])),
            smat_minus_native=dict(difference=float(np.mean([row['smat_minus_native']['difference'] for row in rows])),
                                  ci95=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist())))
jr.atomic_json(args.output/'result.json',dict(complete=True,selection=selection,
    split_code=args.split_code,examples_per_cell=args.examples_per_cell,metrics=metrics,comparisons=comparisons,
    macro_comparisons=macro_comparisons,
    additional_comparisons=additional_comparisons,additional_macro_comparisons=additional_macro_comparisons,
    interval_scope='Paired bootstrap over independent tables, stratified by cell for macro scores; conditional on training seed123, not training-seed uncertainty.'))
print('PREDICTIONS_COMPLETE',str(args.output),flush=True)
for name in ['audit_joint_recall_confirmation.py','collect_joint_recall_confirmation.py']:
    subprocess.run([sys.executable,'-u',str(evaluation_source/name),str(args.output)],check=True)
print('CONFIRMATION_COMPLETE',str(args.output),flush=True)

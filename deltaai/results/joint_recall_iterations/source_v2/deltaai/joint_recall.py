"""Fixed block-context joint recall, paired key reuse, and resumable epoch training.

Block structure follows Zhan et al. (2025); inquiry answer slots are masked.
Five training cells use the existing MQAR 100k+4*20k sampling budget.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from synthetic_memory import make_model, query_logits, auxiliary, evaluate, atomic_json

PAD, BOS, SEP = 0, 1, 2
CONTEXT_BASE, CONTEXT_COUNT = 3, 32
KEY_BASE, KEY_COUNT = 35, 128
VALUE_BASE, VALUES, VOCAB = 163, 16, 179
CELLS = [(2,2,100000), (2,4,20000), (4,4,20000), (4,8,20000), (8,8,20000)]
DATA_SEED = 20260918
LAYOUT = 'block'
VALIDATION_PER_CELL, TEST_PER_CELL = 1000, 2000
ROOT = Path(__file__).resolve().parent/'results/joint_recall'


def cell_name(c,k): return f'c{c}-k{k}'
def length(c,k):
    raw=6*c*k+2 if LAYOUT=='record' else 2+2*c+4*c*k
    return max(64,4*math.ceil(raw/4))


def configure_dataset(root,config_path=None):
    global CELLS,DATA_SEED,LAYOUT,VALIDATION_PER_CELL,TEST_PER_CELL
    path=config_path or root/'dataset_config.json'
    if not path.exists():return
    config=json.loads(path.read_text())
    CELLS=[tuple(map(int,row)) for row in config['cells']]
    DATA_SEED=int(config.get('data_seed',20260918))
    LAYOUT=config.get('layout','block')
    VALIDATION_PER_CELL=int(config.get('validation_per_cell',1000))
    TEST_PER_CELL=int(config.get('test_per_cell',2000))
    assert LAYOUT in ['block','record']
    assert sum(n for _,_,n in CELLS)==180000
    assert all(1<=c<=CONTEXT_COUNT and c*k<=KEY_COUNT and n>0 for c,k,n in CELLS)
    destination=root/'dataset_config.json'
    if destination.exists():assert json.loads(destination.read_text())==config
    else:atomic_json(destination,config)
def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def generate(c,k,n,split):
    """Return paired inputs, all queried targets, and their supervision positions."""
    rng=np.random.default_rng(np.random.SeedSequence([DATA_SEED,split,c,k]))
    unique=np.zeros((n,length(c,k)),dtype=np.uint16)
    shared=np.zeros_like(unique)
    positions=np.empty((n,c*k),dtype=np.uint16)
    targets=np.empty_like(positions)
    for b in range(n):
        contexts=rng.choice(CONTEXT_COUNT,c,replace=False)+CONTEXT_BASE
        pool=rng.choice(KEY_COUNT,c*k,replace=False)+KEY_BASE
        ku=pool.reshape(c,k);ks=np.tile(pool[:k],(c,1))
        values=rng.integers(VALUES,size=(c,k))+VALUE_BASE
        if LAYOUT=='record':
            ctx=np.repeat(contexts,k)
            info=rng.permutation(c*k);query=rng.permutation(c*k)
            half=length(c,k)//2
            for x,keys in [(unique,ku),(shared,ks)]:
                table=np.column_stack((ctx,keys.ravel(),values.ravel()))
                x[b,0]=BOS;x[b,1:1+3*c*k]=table[info].ravel()
                table[:,2]=PAD
                x[b,half]=SEP;x[b,half+1:half+1+3*c*k]=table[query].ravel()
            positions[b]=half+2+3*np.arange(c*k)
            targets[b]=values.ravel()[query]
            continue
        unique[b,0]=shared[b,0]=BOS; offset=1
        for ci in rng.permutation(c):
            unique[b,offset]=shared[b,offset]=contexts[ci];offset+=1
            for ki in rng.permutation(k):
                unique[b,offset:offset+2]=[ku[ci,ki],values[ci,ki]]
                shared[b,offset:offset+2]=[ks[ci,ki],values[ci,ki]];offset+=2
        offset=length(c,k)//2  # Pad both task components equally for existing kernels.
        unique[b,offset]=shared[b,offset]=SEP;offset+=1;q=0
        for ci in rng.permutation(c):
            unique[b,offset]=shared[b,offset]=contexts[ci];offset+=1
            for ki in rng.permutation(k):
                unique[b,offset]=ku[ci,ki];shared[b,offset]=ks[ci,ki]
                positions[b,q]=offset;targets[b,q]=values[ci,ki]
                offset+=2;q+=1  # PAD in answer slots; no answer feedback.
        assert offset<=length(c,k)
    return unique,shared,positions,targets


def prepare(root):
    folder=root/'data';folder.mkdir(parents=True,exist_ok=True)
    manifest_path=folder/'manifest.json'
    if manifest_path.exists():return json.loads(manifest_path.read_text())
    manifest=dict(version=2 if not (root/'dataset_config.json').exists() else 3,
        data_seed=DATA_SEED,vocab=VOCAB,values=VALUES,
        train_examples=sum(n for _,_,n in CELLS),validation_examples=VALIDATION_PER_CELL*len(CELLS),test_examples=TEST_PER_CELL*len(CELLS),
        encoding=('BOS context-blocks SEP permuted inquiry-blocks; masked answer slots' if LAYOUT=='block'
                  else 'BOS shuffled (context,key,value) records SEP shuffled (context,key,PAD) queries'),
        paired_conditions=['unique','shared'],cells=[],files={})
    for c,k,train_n in CELLS:
        name=cell_name(c,k);manifest['cells'].append(dict(name=name,contexts=c,keys=k,records=c*k,length=length(c,k),train_examples=train_n))
        for split,code,n in [('train',1001,train_n),('validation',2001,VALIDATION_PER_CELL),('test',3001,TEST_PER_CELL)]:
            arrays=generate(c,k,n,code)
            for tag,array in zip(['unique','shared','positions','targets'],arrays):
                path=folder/f'{name}-{split}-{tag}.npy'
                np.save(path,array)
                manifest['files'][path.name]=dict(shape=list(array.shape),dtype=str(array.dtype),sha256=sha_file(path))
            print(f'DATA_READY {name} {split} examples={n} length={length(c,k)}',flush=True)
    atomic_json(manifest_path,manifest)
    return manifest


def load_data(root,condition,split):
    out=[]
    for c,k,_ in CELLS:
        stem=root/'data'/f'{cell_name(c,k)}-{split}'
        out.append(tuple(np.load(f'{stem}-{tag}.npy',mmap_mode='r') for tag in [condition,'positions','targets']))
    return out


def batch(data,indices):return tuple(np.asarray(a[indices],dtype=np.int64) for a in data)


@torch.no_grad()
def evaluate_all(model,data,batch_size,folder=None):
    cells={}
    for (c,k,_),arrays in zip(CELLS,data):
        # Evaluation is independent of the training batch permutation.
        metrics,predictions=evaluate(model,tuple(np.asarray(a,dtype=np.int64) for a in arrays),batch_size,return_predictions=True)
        cells[cell_name(c,k)]=metrics
        if folder is not None:
            np.savez_compressed(folder/f'test-{cell_name(c,k)}.npz',predictions=predictions,targets=arrays[2],positions=arrays[1])
    return dict(accuracy=float(np.mean([v['accuracy'] for v in cells.values()])),
                exact_accuracy=float(np.mean([v['exact_accuracy'] for v in cells.values()])),cells=cells)


def run_one(args,condition,family,d,seed,manifest):
    folder=args.root/f'{condition}-{family}-d{d}-w{args.width}-s{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return True
    train=load_data(args.root,condition,'train');valid=load_data(args.root,condition,'validation')
    steps_per_epoch=sum(math.ceil(len(a[0])/args.batch) for a in train)
    recipe=dict(condition=condition,family=family,d=d,width=args.width,seed=seed,
        epochs=args.epochs,batch=args.batch,steps_per_epoch=steps_per_epoch,
        total_steps=args.epochs*steps_per_epoch,examples_per_epoch=manifest['train_examples'],
        example_presentations=args.epochs*manifest['train_examples'],lr=args.lr,weight_decay=.1,
        scheduler='cosine by epoch to zero',precision='BF16 autocast',gradient_clipping=None,
        layers=2,head_dim=16,state_dim=16,vocab=VOCAB,data_seed=DATA_SEED,
        dataset_manifest_sha256=sha_file(args.root/'data/manifest.json'),
        lengths=list(dict.fromkeys(length(c,k) for c,k,_ in CELLS)),validation_examples=manifest['validation_examples'],test_examples=manifest['test_examples'],
        eval_batch=args.batch,early_stopping=False)
    if args.memory_timescale:
        if family!='gdn_current':raise ValueError('Memory-timescale initialization currently applies to GDN only')
        recipe['memory_timescale_initialization']=args.memory_timescale
    recipe_path=folder/'recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text())!=recipe:raise ValueError('Recipe mismatch: '+str(folder))
    atomic_json(recipe_path,recipe)
    torch.set_num_threads(2);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=make_model(family,args.width,d,recipe['lengths'],VOCAB)
    if args.memory_timescale:
        with torch.no_grad():
            for module in model.modules():
                if module.__class__.__name__=='GatedDeltaNet':
                    # At a_proj(u)=0, each scalar gate has alpha=exp(-1/tau).
                    # This only initializes existing parameters; all remain trainable.
                    delta=1/(args.memory_timescale*module.A_log.float().exp())
                    module.dt_bias.copy_(torch.log(torch.expm1(delta)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.1)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    epoch=cursor=steps=0;elapsed=0.;best=-1.
    checkpoint=folder/'checkpoint.pt'
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        epoch,cursor,steps,elapsed,best=[ck[k] for k in ['epoch','cursor','steps','elapsed','best']]
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        for name,module in model.named_modules():
            if hasattr(module,'_steps'):module._steps=ck['module_steps'].get(name,0)
    base=dict(condition=condition,family=family,d=d,width=args.width,seed=seed,
              parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    started=time.monotonic();deadline=started+args.max_minutes*60
    def save():
        temp=checkpoint.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch,cursor=cursor,steps=steps,elapsed=elapsed+time.monotonic()-started,best=best,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            module_steps={n:m._steps for n,m in model.named_modules() if hasattr(m,'_steps')}),temp)
        temp.replace(checkpoint)
    print('START '+json.dumps(dict(base,epoch=epoch,cursor=cursor,steps=steps,recipe=recipe)),flush=True)
    while epoch<args.epochs:
        rng=np.random.default_rng(np.random.SeedSequence([seed,4001,epoch]))
        orders=[rng.permutation(len(a[0])) for a in train]
        batches=[(i,s) for i,a in enumerate(train) for s in range(0,len(a[0]),args.batch)]
        rng.shuffle(batches);assert len(batches)==steps_per_epoch
        model.train()
        for bi in range(cursor,len(batches)):
            ci,start=batches[bi];data=batch(train[ci],orders[ci][start:start+args.batch])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,targets=query_logits(model,data)
                ce=F.cross_entropy(logits.flatten(0,1).float(),targets.flatten())
                loss=ce+auxiliary(model)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss in {folder} at step{steps}')
            loss.backward();optimizer.step();steps+=1;cursor=bi+1
            if steps%100==0:
                status=dict(base,epoch=epoch,batch_in_epoch=cursor,steps=steps,train_loss=ce.item(),elapsed_seconds=elapsed+time.monotonic()-started)
                atomic_json(folder/'status.json',status);print('PROGRESS '+json.dumps(status),flush=True)
            if (args.pilot_steps and steps>=args.pilot_steps) or time.monotonic()>deadline:
                save()
                if args.pilot_steps:
                    metrics=evaluate_all(model,valid,args.batch)
                    atomic_json(folder/'pilot_metrics.json',dict(base,steps=steps,**metrics))
                    print('PILOT '+json.dumps(dict(base,steps=steps,**metrics)),flush=True)
                print('TIME_SLICE_COMPLETE '+str(folder),flush=True);return False
        metrics=evaluate_all(model,valid,args.batch)
        epoch+=1;cursor=0;scheduler.step()
        if metrics['accuracy']>best:
            best=metrics['accuracy']
            temp=folder/'best.tmp';torch.save(dict(model=model.state_dict(),epoch=epoch,steps=steps,metrics=metrics,
                module_steps={n:m._steps for n,m in model.named_modules() if hasattr(m,'_steps')}),temp);temp.replace(folder/'best.pt')
        record=dict(base,epoch=epoch,steps=steps,lr=optimizer.param_groups[0]['lr'],elapsed_seconds=elapsed+time.monotonic()-started,**metrics)
        # One durable row per epoch, including after interrupted evaluation.
        history=folder/'metrics.jsonl'
        existing=[json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
        existing=[r for r in existing if r['epoch']<epoch]+[record]
        history.write_text(''.join(json.dumps(r)+'\n' for r in existing))
        save();atomic_json(folder/'status.json',record);print('EPOCH '+json.dumps(record),flush=True)
    test=load_data(args.root,condition,'test')
    final=evaluate_all(model,test,args.batch,folder)
    best_ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(best_ck['model'])
    for name,module in model.named_modules():
        if hasattr(module,'_steps'):module._steps=best_ck['module_steps'].get(name,0)
    best_folder=folder/'best_test';best_folder.mkdir(exist_ok=True)
    best_test=evaluate_all(model,test,args.batch,best_folder)
    result=dict(base,complete=True,epochs=epoch,steps=steps,examples_seen=epoch*manifest['train_examples'],
                elapsed_seconds=elapsed+time.monotonic()-started,final_test=final,
                best_validation_epoch=best_ck['epoch'],best_validation_accuracy=best,best_test=best_test)
    atomic_json(folder/'result.json',result);print('RESULT '+json.dumps(result),flush=True)
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--conditions',default='shared,unique')
    p.add_argument('--families',default='mamba2,gdn_current')
    p.add_argument('--ds',default='1,3')
    p.add_argument('--seeds',default='123,456,789')
    p.add_argument('--epochs',type=int,default=32)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--memory-timescale',type=float,default=0.)
    p.add_argument('--dataset-config',type=Path)
    p.add_argument('--pilot-steps',type=int,default=0)
    p.add_argument('--max-minutes',type=float,default=105)
    p.add_argument('--worker',type=int,default=0)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    configure_dataset(args.root,args.dataset_config)
    manifest=prepare(args.root)
    if args.prepare_only:return
    cells=[(condition,family,d,seed) for seed in map(int,args.seeds.split(','))
           for condition in args.conditions.split(',') for d in map(int,args.ds.split(',')) for family in args.families.split(',')]
    started=time.monotonic();budget=args.max_minutes
    for i,(condition,family,d,seed) in enumerate(cells):
        if i%args.workers!=args.worker:continue
        if condition not in ['shared','unique']:raise ValueError(condition)
        args.max_minutes=budget-(time.monotonic()-started)/60
        if args.max_minutes<2:return
        complete=run_one(args,condition,family,d,seed,manifest)
        gc.collect();torch.cuda.empty_cache()
        if not complete and not args.pilot_steps:return
    print('CAMPAIGN_COMPLETE' if not args.pilot_steps else 'PILOT_COMPLETE',flush=True)


if __name__=='__main__':main()

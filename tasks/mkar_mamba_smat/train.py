"""Original MKAR training/evaluation protocol, with resumable independent k fits."""
import argparse
import fcntl
import json
import signal
import time
import numpy as np
from common import ROOT, PROTOCOL as P, bootstrap, atomic_json, verify_sources

bootstrap()
import torch
from torch.nn import functional as F
from mkar import MKAR, evaluate
from campaign_model import Model


def task_for(k, seed, device='cuda'):
    return MKAR(P['length'],P['boundary'],P['records'],k,P['answers'],P['vocab'],device,seed=seed)


def loss_for(model, data):
    x,pay,target,rows,_,_ = data
    out = model(x,pay)
    pick = out.gather(1,rows.unsqueeze(-1).expand(-1,-1,target.shape[-1]))
    loss = F.binary_cross_entropy_with_logits(pick,target)
    return loss, loss+model.auxiliary()


def fit(job, k, deadline, stop, source_hash):
    seed, config = job['seed'], job['config']
    folder = ROOT/'runs'/job['name']/f'k{k}'
    folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    torch.manual_seed(seed)
    np.random.seed(seed)
    model=Model(config,seed).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
    task=task_for(k,seed)
    recipe=dict(job=job,k=k,protocol=P,source_manifest_sha256=source_hash,model=model.audit(),
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__)
    if (folder/'recipe.json').exists():
        old=json.loads((folder/'recipe.json').read_text())
        assert all(old[tag]==recipe[tag] for tag in ('job','k','protocol','source_manifest_sha256','model'))
    else:atomic_json(folder/'recipe.json',recipe)
    step=0;elapsed=0.;last_eval=None
    checkpoint=folder/'checkpoint.pt'
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['source_hash']==source_hash
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        step,elapsed,last_eval=ck['step'],ck['elapsed'],ck['last_eval']
        task.rng.bit_generator.state=ck['data_rng']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        for name,m in model.named_modules():
            if hasattr(m,'_steps'):m._steps=ck['module_steps'].get(name,0)
        del ck
    started=time.monotonic()
    def save():
        tmp=checkpoint.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
            elapsed=elapsed+time.monotonic()-started,last_eval=last_eval,
            data_rng=task.rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),source_hash=source_hash,
            module_steps={n:m._steps for n,m in model.named_modules() if hasattr(m,'_steps')}),tmp)
        tmp.replace(checkpoint)
    print('START',job['name'],'k',k,'step',step,flush=True)
    model.train()
    while step<P['steps']:
        ce,loss=loss_for(model,task.batch(P['batch']))
        assert torch.isfinite(loss), (job['name'],k,step,'nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),P['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();step+=1
        if step==1 or step%100==0:
            status=dict(name=job['name'],k=k,step=step,total_steps=P['steps'],
                epoch=step/P['steps_per_epoch'],train_bce=float(ce.detach()),
                gradient_norm=float(norm),elapsed_seconds=elapsed+time.monotonic()-started)
            atomic_json(folder/'status.json',status)
            print('PROGRESS',json.dumps(status),flush=True)
        if step%P['eval_every']==0:
            # The original function and the same training RNG stream are used unchanged.
            last_eval=evaluate(model,task,P['boundary'],P['batch'],P['eval_batches'],False)
            record=dict(step=step,train_bce=float(ce.detach()),evaluation=last_eval)
            with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('EVAL',job['name'],k,json.dumps(record),flush=True)
        if step%P['steps_per_epoch']==0 or step%P['eval_every']==0 or stop[0] or time.monotonic()>deadline:
            save()
        if stop[0] or time.monotonic()>deadline:
            return None
    assert last_eval is not None
    result=dict(name=job['name'],config=config,seed=seed,k=k,complete=True,steps=step,
        epochs=32,evaluation=last_eval,selection='final step',source_manifest_sha256=source_hash,
        model=model.audit(),elapsed_seconds=elapsed+time.monotonic()-started)
    atomic_json(result_path,result)
    atomic_json(folder/'status.json',dict(state='complete',steps=step,k=k))
    return result


def main(index,max_minutes):
    torch.set_num_threads(2)
    source_hash=verify_sources()
    validation=json.loads((ROOT/'validated.json').read_text())
    assert validation['passed'] and validation['source_manifest_sha256']==source_hash
    job=json.loads((ROOT/'tasks.json').read_text())[index]
    folder=ROOT/'runs'/job['name'];folder.mkdir(parents=True,exist_ok=True)
    stop=[False]
    def interrupted(*_):stop[0]=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGUSR1,interrupted)
    deadline=time.monotonic()+60*max_minutes
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results=[]
        for k in P['requested_sizes']:
            result=fit(job,k,deadline,stop,source_hash)
            if result is None:return 75
            results.append(result)
            torch.cuda.empty_cache()
        atomic_json(folder/'result.json',dict(name=job['name'],complete=True,cells=results))
    print('COMPLETE',job['name'],flush=True)
    return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True)
    ap.add_argument('--max-minutes',type=float,default=700);args=ap.parse_args()
    raise SystemExit(main(args.index,args.max_minutes))

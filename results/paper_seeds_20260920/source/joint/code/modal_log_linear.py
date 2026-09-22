"""Six Log-Linear MQAR runs. GPU validation precedes all training dispatch."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import modal

LOCAL=Path(__file__).resolve().parent
app=modal.App('mqar-log-linear-s123')
volume=modal.Volume.from_name('mqar-log-linear-s123',create_if_missing=True)
if modal.is_local():
    from modal_mqar_four_reads import image
    for name in ('zoo_log_linear.py','zoo_log_linear_configs.py','test_log_linear_gpu.py',
                 'zoo_mqar_resume.py','zoo_mqar_interleave.py'):
        image=image.add_local_file(LOCAL/name,'/opt/log-linear/'+name)
    image=image.add_local_file(LOCAL/'results/mqar_log_linear/upstream/hattention/base.py',
                               '/opt/log-linear/upstream_base.py')
else:
    image=modal.Image.debian_slim()

def environment():
    return dict(os.environ,PYTHONPATH='/opt/log-linear:'+os.environ['PYTHONPATH'],
        PYTHONSAFEPATH='1',TRITON_CACHE_DIR='/ll-results/triton-cache')

def execute(cmd,env,root,name):
    with (root/name).open('a') as log:
        child=subprocess.Popen(cmd,cwd='/opt/smat/the GPU cluster',env=env,stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True)
        for line in child.stdout:
            log.write(line);log.flush()
            if any(marker in line for marker in ('PASS','CHECKPOINT','SOURCE_AUDIT','TRAINING','OPTIMIZER_CONFIG','INTERLEAVED')):
                print(line,end='',flush=True)
            if 'CHECKPOINT epoch=' in line:
                epoch=int(line.split('epoch=',1)[1].split()[0])
                snapshots=root/'checkpoints';snapshots.mkdir(exist_ok=True)
                shutil.copy2(root/f'w{env["ZOO_DM"]}-d1.pt',snapshots/f'epoch{epoch:02d}.pt')
                volume.commit()
        code=child.wait()
    volume.commit()
    if code:
        print((root/name).read_text()[-10000:],flush=True)
        raise RuntimeError(f'{name} exit {code}')

@app.function(image=image,gpu='H100',cpu=4,memory=32768,timeout=1800,volumes={'/ll-results':volume})
def validate():
    root=Path('/ll-results/validation');root.mkdir(parents=True,exist_ok=True)
    execute([sys.executable,'-P','-u','/opt/log-linear/test_log_linear_gpu.py'],environment(),root,'gpu.log')
    sha=hashlib.sha256(Path('/opt/log-linear/zoo_log_linear.py').read_bytes()).hexdigest()
    (root/'passed.json').write_text(json.dumps(dict(sha256=sha,passed=True)))
    volume.commit()
    return dict(sha256=sha,passed=True)

@app.function(image=image,gpu='H100',cpu=4,memory=32768,timeout=21600,
              max_containers=6,volumes={'/ll-results':volume})
def train(family:str,width:int):
    assert family in ('gdn','mamba2') and width in (16,32,64)
    volume.reload()
    root=Path('/ll-results')/f'{family}-w{width}';root.mkdir(parents=True,exist_ok=True)
    source=Path('/opt/log-linear/zoo_log_linear.py')
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(Path('/ll-results/validation/passed.json').read_text())['sha256']==sha
    manifest=dict(family=family,width=width,seed=123,lr=.003 if family=='gdn' and width==64 else .01,
        layers=2,epochs=32,head_dim=16,state_dim=16,heads=(1 if width==16 else 2) if family=='gdn' else width//8,
        backend='dense-pytorch-exact-log-linear',hierarchy='base2-WEAK',levels=9,
        upstream_commit='7f8644159c1406fae1ad863829a5b3a4fbf63022',sha256=sha,
        lambda_mode='softplus(L * l_proj(u))',weight_decay=.1,precision='float32',
        note='Matched Zoology backbone; MQAR levels9 rather than unused upstream LM levels15; no epoch gates')
    manifest_path=root/'manifest.json'
    if manifest_path.exists(): assert json.loads(manifest_path.read_text())==manifest
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    source_dir=root/'sources';source_dir.mkdir(exist_ok=True)
    for src in Path('/opt/log-linear').glob('*.py'): shutil.copy2(src,source_dir/src.name)
    volume.commit()
    env=dict(environment(),LL_FAMILY=family,LL_SOURCE_SHA256=sha,ZOO_DM=str(width),ZOO_DS='1',
             ZOO_EPOCHS='32',MQAR_RUN_DIR=str(root),MQAR_MAX_MINUTES='340')
    env.pop('ZOO_SMOKE',None)
    execute([sys.executable,'-P','-u','-m','zoology.launch','/opt/log-linear/zoo_log_linear_configs.py'],env,root,'train.log')
    status=json.loads((root/f'w{width}-d1.json').read_text())
    (root/'result.json').write_text(json.dumps(status,indent=2)+'\n');volume.commit()
    return dict(family=family,width=width,**status)

@app.local_entrypoint()
def sweep():
    passed=validate.remote()
    assert passed['passed']
    runs=[]
    for family in ('gdn','mamba2'):
        for width in (16,32,64):
            call=train.spawn(family,width)
            runs.append(dict(family=family,width=width,call_id=call.object_id,
                             lr=.003 if family=='gdn' and width==64 else .01))
    path=LOCAL/'results/mqar_log_linear/campaign.json'
    path.write_text(json.dumps(dict(seed=123,runs=runs,validation=passed),indent=2)+'\n')
    print('LAUNCHED '+path.read_text(),flush=True)
    for run in runs:
        print('RESULT '+json.dumps(modal.FunctionCall.from_id(run['call_id']).get()),flush=True)

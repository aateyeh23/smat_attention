"""Four matched-width PG19 arms; short resumable full-model GPU pilots first."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import modal
LOCAL=Path(__file__).resolve().parent
app=modal.App('pg19-w768-rank64')
data=modal.Volume.from_name('smat-pg19-scale-data-v1')
volume=modal.Volume.from_name('pg19-w768-rank64',create_if_missing=True)
FILES=('smat_write_hash_triton.py','lm/test_write_hash_fused_gpu.py','content_addr.py','smat_read_triton.py','smat_write_hash_grad.py','smat_address_reads.py',
 'smat_gdn_transport.py','zoo_gdn_transport.py','zoo_smat_gdn.py','zoo_smat_mixer.py',
 'lm/gdn_smat_scale.py','lm/gdn_smat_transport_scale.py','lm/mamba_smat_scale.py',
 'lm/train_pg19_scale.py','lm/test_pg19_rank64_gpu.py','lm/run_pg19_rank64.py')
if modal.is_local():
    from modal_mqar_four_reads import image
    for name in FILES:image=image.add_local_file(LOCAL/name,'/opt/pg19-rank64/'+name)
else:image=modal.Image.debian_slim()

@app.function(image=image,gpu='H100',cpu=4,memory=49152,timeout=43200,
              max_containers=4,volumes={'/data':data,'/checkpoints':volume})
def run(family:str,d:int,pilot:bool=True,fused_write:bool=True,batch:int=0,activation_checkpointing:bool=False):
    assert family in ('gdn','mamba2') and d in (3,4)
    batch=batch or (2 if family=='gdn' and d==3 else 1)
    assert batch in (1,2,4,8), 'Microbatch must divide global batch8'
    data.reload();volume.reload()
    root=Path('/checkpoints/seed123')
    if fused_write:root=root/f'fusedwrite-b{batch}-ac{int(activation_checkpointing)}'
    root=root/f'{family}-d{d}';root.mkdir(parents=True,exist_ok=True)
    sources={n:hashlib.sha256(Path('/opt/pg19-rank64',n).read_bytes()).hexdigest() for n in FILES}
    heads,head_dim=(6,128) if family=='gdn' else (24,64)
    variant='updated_transport' if family=='gdn' else 'mamba2_fixed_write'
    manifest=dict(family=family,d=d,width=768,layers=16,ffn_width=2048,
        heads=heads,head_dim=head_dim,read_rank=64,length=16384,seed=123,
        target_tokens=2_000_000_000,max_hours=11.8,lr=.0003,batch=batch,
        global_batch_rows=8,activation_checkpointing=activation_checkpointing,write_hash_neighbor_grad=True,
        write_hash_backend='triton' if fused_write else 'torch',
        source_sha256=sources)
    path=root/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Changed experiment manifest'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    for name in FILES:
        dst=root/'sources'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path('/opt/pg19-rank64',name),dst)
    volume.commit()
    env=dict(os.environ,PYTHONPATH='/opt/pg19-rank64/lm:/opt/pg19-rank64:'+os.environ['PYTHONPATH'],
        PYTHONSAFEPATH='1',PG19_MEMORY_VARIANT=variant,PG19_EXPECTED_SOURCES=json.dumps(sources),
        TRITON_CACHE_DIR='/tmp/pg19-triton-cache',
        SMAT_WRITE_HASH_BACKEND='triton' if fused_write else 'torch')
    def execute(args,name):
        last=time.monotonic()
        with (root/name).open('a') as fp:
            child=subprocess.Popen([sys.executable,'-P','-u','/opt/pg19-rank64/lm/run_pg19_rank64.py',*args],
                cwd='/opt/pg19-rank64',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:
                fp.write(line);fp.flush();print(f'{family}-d{d} '+line,end='',flush=True)
                if '"event": "checkpoint"' in line or time.monotonic()-last>45:
                    volume.commit();last=time.monotonic()
            code=child.wait()
        volume.commit()
        if code:raise RuntimeError(f'{family}-d{d} {name} exit{code}')
    if pilot:
        execute(['validate','--d',str(d)],'validation.log')
        (root/'validated.json').write_text(json.dumps(dict(passed=True,sources=sources)))
        volume.commit()
    else:
        assert json.loads((root/'validated.json').read_text())['sources']==sources
    args=['train','--d',str(d),'--data','/data/pg19-16k-2b','--output',str(root),
        '--width','768','--layers','16','--ffn-width','2048','--heads',str(heads),
        '--head-dim',str(head_dim),'--read-rank','64','--batch',str(batch),'--global-batch-rows','8',
        '--log-every','1' if pilot else '10']
    if activation_checkpointing:args+=['--activation-checkpointing']
    if pilot:args+=['--max-steps','4']
    execute(args,'train.log')
    status=json.loads((root/'status.json').read_text())
    if pilot:
        rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
        steady=[r['tokens_per_second'] for r in rows if r['event']=='train' and r['step']>=2]
        result=dict(family=family,d=d,mean_tokens_per_second=sum(steady)/len(steady),status=status)
        result['hours_for_2b']=2e9/result['mean_tokens_per_second']/3600
        (root/'pilot.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
        print('PILOT '+json.dumps(result),flush=True)
        return result
    return dict(family=family,d=d,**status)

@app.function(timeout=46800)
def sweep(pilot:bool=True,fused_write:bool=True,batch:int=0,activation_checkpointing:bool=False):
    calls={f'{f}-d{d}':run.spawn(f,d,pilot,fused_write,batch,activation_checkpointing) for f in ('gdn','mamba2') for d in (3,4)}
    print('CALLS '+json.dumps({k:c.object_id for k,c in calls.items()}),flush=True)
    results={}
    for key,call in calls.items():
        try:results[key]=call.get()
        except Exception as error:results[key]=dict(error=str(error))
    print('RESULTS '+json.dumps(results),flush=True)
    return results

@app.function(image=image,gpu='H100',cpu=4,memory=16384,timeout=1200)
def validate_write_grad():
    env=dict(os.environ,PYTHONPATH='/opt/pg19-rank64/lm:/opt/pg19-rank64:'+os.environ['PYTHONPATH'],PYTHONSAFEPATH='1')
    subprocess.run([sys.executable,'-P','-u','/opt/pg19-rank64/lm/test_write_hash_fused_gpu.py'],env=env,check=True)
    return 'fused write-gradient checks passed'

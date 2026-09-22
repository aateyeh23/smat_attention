"""Fresh d=3/4 PG19 pretraining using the updated transport model."""
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
app=modal.App('pg19-updated-transport-d34')
data=modal.Volume.from_name('smat-pg19-scale-data-v1')
volume=modal.Volume.from_name('pg19-updated-transport-d34',create_if_missing=True)
FILES=('content_addr.py','smat_read_triton.py','smat_write_hash_grad.py','smat_address_reads.py',
       'smat_gdn_transport.py','zoo_gdn_transport.py','zoo_smat_gdn.py',
       'lm/gdn_smat_scale.py','lm/gdn_smat_transport_scale.py','lm/train_pg19_scale.py',
       'lm/test_pg19_transport_gpu.py','lm/run_pg19_transport.py')
if modal.is_local():
    from modal_mqar_four_reads import image
    for name in FILES:image=image.add_local_file(LOCAL/name,'/opt/pg19-updated/'+name)
else:image=modal.Image.debian_slim()

@app.function(image=image,gpu='H100',cpu=4,memory=49152,timeout=43200,
              max_containers=2,volumes={'/data':data,'/checkpoints':volume})
def train(d:int):
    assert d in (3,4)
    data.reload();volume.reload()
    root=Path('/checkpoints/seed123')/f'd{d}';root.mkdir(parents=True,exist_ok=True)
    sources={name:hashlib.sha256(Path('/opt/pg19-updated',name).read_bytes()).hexdigest() for name in FILES}
    manifest=dict(d=d,variant='updated_transport',seed=123,tokens=2_000_000_000,
        width=1024,layers=16,ffn_width=2816,length=16384,heads=2,head_dim=64,
        lr=.0003,batch=3,global_batch_rows=8,activation_checkpointing=True,
        max_hours=11.8,write_hash_neighbor_grad=True,transport_scalar_decay=True,
        detach_write_hash_input=True,memory_plant=False,memory_incidence_rescale=False,
        read_k=4,source_sha256=sources)
    p=root/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest
    p.write_text(json.dumps(manifest,indent=2)+'\n')
    for name in FILES:
        dst=root/'sources'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path('/opt/pg19-updated',name),dst)
    volume.commit()
    env=dict(os.environ,PYTHONPATH='/opt/pg19-updated/lm:/opt/pg19-updated:'+os.environ['PYTHONPATH'],
             PYTHONSAFEPATH='1',PG19_MEMORY_VARIANT='updated_transport',
             PG19_EXPECTED_SOURCES=json.dumps(sources),TRITON_CACHE_DIR='/checkpoints/triton-cache')
    def execute(args,name):
        last=time.monotonic()
        with (root/name).open('a') as fp:
            child=subprocess.Popen([sys.executable,'-P','-u','/opt/pg19-updated/lm/run_pg19_transport.py',*args],
                cwd='/opt/pg19-updated',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:
                fp.write(line);fp.flush();print(f'd{d} '+line,end='',flush=True)
                if '"event": "checkpoint"' in line or time.monotonic()-last>45:
                    volume.commit();last=time.monotonic()
            code=child.wait()
        volume.commit()
        if code:raise RuntimeError(f'd{d}: {name} failed exit{code}')
    execute(['validate','--d',str(d)],'validation.log')
    (root/'validated.json').write_text(json.dumps(dict(passed=True,sources=sources)))
    volume.commit()
    execute(['train','--d',str(d),'--data','/data/pg19-16k-2b','--output',str(root),
        '--batch','3','--global-batch-rows','8','--activation-checkpointing'], 'train.log')
    return json.loads((root/'status.json').read_text())

@app.function(timeout=46800)
def sweep():
    calls={d:train.spawn(d) for d in (3,4)}
    print('CALLS '+json.dumps({d:c.object_id for d,c in calls.items()}),flush=True)
    return {d:c.get() for d,c in calls.items()}

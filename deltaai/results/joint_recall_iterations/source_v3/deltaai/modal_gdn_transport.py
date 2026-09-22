"""Validate and train only the requested w32/d3 transport experiment on one H100."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import modal

LOCAL=Path(__file__).resolve().parent
app=modal.App('smat-gdn-transport-w32-d3')
volume=modal.Volume.from_name('smat-gdn-transport-w32-d3',create_if_missing=True)
if modal.is_local():
    from modal_mqar_width_sweeps import image as cached_image
    image=cached_image
    for filename in ('smat_gdn_transport.py','zoo_gdn_transport.py','zoo_gdn_transport_configs.py',
                     'zoo_gdn_transport_neighbor_configs.py','content_addr.py',
                     'smat_write_hash_grad.py','smat_address_reads.py','test_gdn_transport_gpu.py'):
        image=image.add_local_file(LOCAL/filename,'/opt/transport/'+filename)
else:
    image=modal.Image.debian_slim()


def execute(command,path):
    env=dict(os.environ,PYTHONPATH='/opt/transport:'+os.environ['PYTHONPATH'],
             PYTHONSAFEPATH='1',
             MQAR_RUN_DIR=str(path.parent),MQAR_MAX_MINUTES='165',
             ZOO_DM='32',ZOO_DS='3',ZOO_EPOCHS='32',MQAR_FAMILY='gdn')
    env.pop('ZOO_SMOKE',None)
    previous=time.monotonic()
    with path.open('a') as log:
        child=subprocess.Popen(command,cwd='/opt/smat/deltaai',env=env,stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True)
        for line in child.stdout:
            log.write(line);log.flush()
            if command[-1].endswith('test_gdn_transport_gpu.py') or 'CHECKPOINT' in line or 'Valid Epoch' in line:
                print(line,end='',flush=True)
            if 'CHECKPOINT' in line or time.monotonic()-previous>60:
                volume.commit();previous=time.monotonic()
        code=child.wait()
    volume.commit()
    if code:
        print(path.read_text()[-8000:],flush=True)
        raise RuntimeError(f'Process failed ({code}); log: {path}')


@app.function(image=image,gpu='H100',cpu=4,memory=32768,timeout=10800,
              volumes={'/transport-results':volume})
def run(neighbor_write_grad: bool = False):
    volume.reload()
    root=Path('/transport-results/seed123')
    config='zoo_gdn_transport_configs.py'
    if neighbor_write_grad:
        root=root/'neighbor_write_grad'
        config='zoo_gdn_transport_neighbor_configs.py'
    root.mkdir(exist_ok=True,parents=True)
    execute([sys.executable,'-u','/opt/transport/test_gdn_transport_gpu.py'],root/'gpu-validation.log')
    (root/'validated.json').write_text(json.dumps(dict(gpu_checks=True)))
    volume.commit()
    print(f'GPU_VALIDATED: width32 d3, lr=0.01 seed=123; neighbor_write_grad={neighbor_write_grad}',flush=True)
    execute([sys.executable,'-u','-m','zoology.launch','/opt/transport/'+config],root/'train.log')
    status=json.loads((root/'w32-d3.json').read_text())
    print('FINAL '+json.dumps(status),flush=True)
    return status

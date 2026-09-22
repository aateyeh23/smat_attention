"""One candidate per invocation, checkpointed strict GDN epoch gates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import modal

LOCAL=Path(__file__).resolve().parent
app=modal.App('smat-gdn-w32-d3-iterations')
volume=modal.Volume.from_name('smat-gdn-w32-d3-iterations',create_if_missing=True)
if modal.is_local():
    from modal_gdn_transport import image as cached_image
    image=cached_image
    for name in ('zoo_gdn_gated_trainer.py','zoo_gdn_iteration_configs.py','test_gdn_iteration_gpu.py'):
        image=image.add_local_file(LOCAL/name,'/opt/iterate/'+name)
    image=image.add_local_file(LOCAL/'results/mqar_width_sweeps/gdn/w32-d1-history.jsonl',
                               '/opt/iterate/gdn-baseline.jsonl')
else:
    image=modal.Image.debian_slim()

VARIANTS={
    '01_no_scalar_decay':dict(name='no_scalar_decay',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False)),
    '02_isolated_hash':dict(name='isolated_hash',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,detach_write_hash_input=True)),
    '03_content_only':dict(name='content_only',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False)),
    '04_shared_address':dict(name='shared_address',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_read_address=True,share_hash_across_lengths=True)),
    '05_shared_hash':dict(name='shared_hash',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        share_hash_across_lengths=True)),
    '06_tied_content':dict(name='tied_content',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        transport_tied_features=True)),
    '07_incidence_scale':dict(name='incidence_scale',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True)),
    '08_orthogonal_hash':dict(name='orthogonal_hash',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,orthogonal_hash=True)),
    '08b_orthogonal_hash':dict(name='orthogonal_hash_cayley',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,orthogonal_hash=True)),
    '09_gdn_key_hash':dict(name='gdn_key_hash',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,memory_hash_features='gdn_keys')),
    '10_address_reader':dict(name='address_reader',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,memory_read_address=True)),
    '11_symmetric_write_grad':dict(name='symmetric_write_grad',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,symmetric_write_grad=True)),
    '12_read_exploration':dict(name='read_exploration',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,memory_read_noise=1.)),
    '13_joint_hash_balance':dict(name='joint_hash_balance',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,joint_hash_balance=True)),
    '14_tied_additive':dict(name='tied_additive',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,transport_tied_features=True,
        transport_enabled=False)),
    '15_aligned_additive':dict(name='aligned_additive',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,transport_tied_features=True,
        transport_enabled=False,memory_read_address=True)),
}


@app.function(image=image,gpu='H100',cpu=4,memory=32768,timeout=10800,
              volumes={'/iterate-results':volume},max_containers=1)
def run(trial: str='01_no_scalar_decay'):
    if trial not in VARIANTS: raise ValueError(trial)
    volume.reload()
    root=Path('/iterate-results')/trial;root.mkdir(parents=True,exist_ok=True)
    manifest=dict(trial=trial,**VARIANTS[trial],width=32,d=3,heads=2,head_dim=16,
                  state_dim=16,layers=2,seed=123,epochs=32,learning_rate=.01,
                  gate='strictly beat baseline at epochs 4,6,8,...,32')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=dict(os.environ,PYTHONPATH='/opt/iterate:/opt/transport:'+os.environ['PYTHONPATH'],
             GDN_VARIANT_JSON=json.dumps(VARIANTS[trial]),GDN_BASELINE_HISTORY='/opt/iterate/gdn-baseline.jsonl',
             MQAR_RUN_DIR=str(root),MQAR_MAX_MINUTES='165',ZOO_DM='32',ZOO_DS='3',
             ZOO_EPOCHS='32',MQAR_FAMILY='gdn',TRITON_CACHE_DIR='/iterate-results/triton-cache')
    env.pop('ZOO_SMOKE',None)
    def execute(command,name):
        with (root/name).open('a') as log:
            child=subprocess.Popen(command,cwd='/opt/smat/deltaai',env=env,
                                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:
                log.write(line);log.flush()
                if 'CHECKPOINT' in line or 'GATE ' in line or 'PASS ' in line:
                    print(line,end='',flush=True)
                if 'CHECKPOINT epoch=' in line:
                    epoch=int(line.split('epoch=',1)[1].split()[0])
                    snapshots=root/'checkpoints';snapshots.mkdir(exist_ok=True)
                    shutil.copy2(root/'w32-d3.pt',snapshots/f'epoch{epoch:02d}.pt')
                    volume.commit()
            code=child.wait()
        volume.commit()
        if code:
            print((root/name).read_text()[-9000:],flush=True)
            raise RuntimeError(f'{name}: exit {code}')
    execute([sys.executable,'-u','/opt/iterate/test_gdn_iteration_gpu.py'],'gpu-validation.log')
    execute([sys.executable,'-u','-m','zoology.launch','/opt/iterate/zoo_gdn_iteration_configs.py'],'train.log')
    decision=json.loads((root/'decision.json').read_text())
    status=json.loads((root/'w32-d3.json').read_text())
    result=dict(**decision,complete=status['complete'],trial=trial)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    print('RESULT '+json.dumps(result),flush=True)
    return result

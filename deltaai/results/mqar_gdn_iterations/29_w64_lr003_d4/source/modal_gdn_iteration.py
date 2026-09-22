"""One candidate per invocation, checkpointed strict GDN epoch gates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import hashlib
from datetime import datetime, timezone
import modal

LOCAL=Path(__file__).resolve().parent
app=modal.App('smat-gdn-w32-d3-iterations')
volume=modal.Volume.from_name('smat-gdn-w32-d3-iterations',create_if_missing=True)
if modal.is_local():
    from modal_gdn_transport import image as cached_image
    image=cached_image
    for name in ('zoo_gdn_gated_trainer.py','zoo_gdn_iteration_configs.py',
                 'test_gdn_iteration_gpu.py','zoo_mqar_resume.py','audit_gdn_sources.py'):
        image=image.add_local_file(LOCAL/name,'/opt/iterate/'+name)
    image=image.add_local_file(LOCAL/'results/mqar_width_sweeps/gdn/w32-d1-history.jsonl',
                               '/opt/iterate/gdn-baseline.jsonl')
    image=image.add_local_file(LOCAL/'results/mqar_width_sweeps/gdn/w64-d1-history.jsonl',
                               '/opt/iterate/gdn-baseline-w64.jsonl')
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
    '16_periodic_hash':dict(name='periodic_hash',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,periodic_hash=True)),
    '17_profile_delta':dict(name='profile_delta',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,transport_tied_features=True,
        transport_enabled=False,profile_delta=True)),
    '18_aligned_profile_delta':dict(name='aligned_profile_delta',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,transport_tied_features=True,
        transport_enabled=False,profile_delta=True,memory_read_address=True)),
    '19_clipped_transport':dict(name='clipped_transport',max_grad_norm=1.,kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True)),
    '20_concurrent_profile_delta':dict(name='concurrent_profile_delta',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True,transport_tied_features=True,
        transport_enabled=False,profile_delta=True,memory_read_address=True,
        memory_concurrent_reads=True)),
    '21_slow_routing':dict(name='slow_routing',routing_lr_scale=.1,kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True)),
    '22_verified_transport':dict(name='verified_transport',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=False,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=True)),
    '23_decay_no_rescale':dict(name='decay_no_rescale',kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=True,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=False)),
    '24_decay_no_rescale_d2':dict(name='decay_no_rescale',d=2,kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=True,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=False)),
    '25_decay_no_rescale_d4':dict(name='decay_no_rescale',d=4,kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=True,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=False)),
    '26_w64_decay_no_rescale_d2':dict(name='decay_no_rescale',d=2,kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=True,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=False)),
    '27_w64_decay_no_rescale_d3':dict(name='decay_no_rescale',d=3,kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=True,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=False)),
    '28_w64_decay_no_rescale_d4':dict(name='decay_no_rescale',d=4,kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=True,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=False)),
    '29_w64_lr003_d4':dict(name='decay_no_rescale_lr003',d=4,lr=.003,kwargs=dict(
        write_hash_neighbor_grad=True,transport_scalar_decay=True,
        detach_write_hash_input=True,memory_plant=False,
        memory_incidence_rescale=False)),
    '30_w64_lr003_gdn':dict(name='baseline_lr003',d=1,lr=.003,baseline=True,kwargs={}),
}


@app.function(image=image,gpu='H100',cpu=4,memory=32768,timeout=10800,
              volumes={'/iterate-results':volume},max_containers=1)
def run(trial: str='23_decay_no_rescale', continue_below_baseline: bool=False,
        width: int=32):
    if trial not in VARIANTS: raise ValueError(trial)
    if width not in (32, 64): raise ValueError('Supported model widths:32,64')
    if ('_w64_' in trial) != (width == 64):
        raise ValueError('Trial name and requested model width disagree')
    variant = dict(VARIANTS[trial])
    if continue_below_baseline:
        variant['stop_on_failed_gate'] = False
    d = variant.get('d', 3)
    stem = f'w{width}-d{d}'
    volume.reload()
    root=Path('/iterate-results')/trial;root.mkdir(parents=True,exist_ok=True)
    manifest=dict(trial=trial,**variant,width=width,heads=2,head_dim=16,
                  state_dim=16,layers=2,seed=123,epochs=32,learning_rate=variant.get('lr', .01),
                  gate='strictly beat baseline at epochs 4,6,8,...,32')
    manifest['d'] = d
    if continue_below_baseline:
        manifest['gate'] = 'log baseline comparisons; train through epoch32 regardless of margin'
    if 'lr' in variant:
        manifest['gate'] += '; logged reference is historical GDN lr0.01; compare lr0.003 arms separately'
    if (root/(stem+'.pt')).exists():
        previous = json.loads((root/'manifest.json').read_text())
        policy_fields = {'stop_on_failed_gate', 'gate'} if continue_below_baseline else set()
        if ({k:v for k,v in previous.items() if k not in policy_fields}
                != {k:v for k,v in manifest.items() if k not in policy_fields}):
            raise ValueError('Cannot resume with a changed model or training manifest')
        status = json.loads((root/(stem+'.json')).read_text())
        resume_rejected = (continue_below_baseline and status['complete']
                           and status['next_epoch'] < manifest['epochs']
                           and status['metrics'].get('gate/failed', 0) > .5)
        if not status['complete'] or resume_rejected:
            archive = root/'resume-history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            archive.mkdir(parents=True)
            for name in ('manifest.json','expected-sources.json','gpu-validation-sources.json',
                         'training-sources.json',stem+'.json','decision.json','result.json'):
                if (root/name).exists(): shutil.copy2(root/name,archive/name)
            if resume_rejected and (root/'result.json').exists():
                (root/'result.json').unlink()
            print(f'RESUME_REQUEST epoch={status["next_epoch"]} {stem}',flush=True)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    source_paths = {name:'/opt/transport/'+name+'.py' for name in (
        'content_addr','zoo_gdn_transport','smat_gdn_transport',
        'smat_address_reads','smat_write_hash_grad')}
    source_paths.update({name:'/opt/iterate/'+name+'.py' for name in (
        'zoo_gdn_gated_trainer','zoo_mqar_resume','audit_gdn_sources')})
    source_paths.update({name:'/opt/smat/deltaai/'+name+'.py' for name in (
        'zoo_smat_gdn','smat_delta_pool','zoo_mqar_interleave')})
    expected_sources = {name:dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                        for name,path in source_paths.items()}
    (root/'expected-sources.json').write_text(json.dumps(expected_sources,indent=2)+'\n')
    baseline = '/opt/iterate/gdn-baseline.jsonl' if width == 32 else '/opt/iterate/gdn-baseline-w64.jsonl'
    env=dict(os.environ,PYTHONPATH='/opt/iterate:/opt/transport:'+os.environ['PYTHONPATH'],
             GDN_VARIANT_JSON=json.dumps(variant),GDN_BASELINE_HISTORY=baseline,
             MQAR_RUN_DIR=str(root),MQAR_MAX_MINUTES='165',ZOO_DM=str(width),ZOO_DS=str(d),
             ZOO_EPOCHS='32',MQAR_FAMILY='gdn',TRITON_CACHE_DIR='/iterate-results/triton-cache',
             PYTHONSAFEPATH='1',GDN_EXPECTED_SOURCES_JSON=json.dumps(expected_sources))
    if continue_below_baseline:
        env['MQAR_RESUME_REJECTED'] = '1'
    env.pop('ZOO_SMOKE',None)
    def execute(command,name):
        with (root/name).open('a') as log:
            child=subprocess.Popen(command,cwd='/opt/smat/deltaai',env=env,
                                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:
                log.write(line);log.flush()
                if any(marker in line for marker in ('CHECKPOINT','GATE ','PASS ','GRADIENT_CLIP ','OPTIMIZER_GROUPS ','OPTIMIZER_CONFIG ','SOURCE_AUDIT ','RESUMED ','TRAINING ')):
                    print(line,end='',flush=True)
                if 'CHECKPOINT epoch=' in line:
                    epoch=int(line.split('epoch=',1)[1].split()[0])
                    snapshots=root/'checkpoints';snapshots.mkdir(exist_ok=True)
                    shutil.copy2(root/(stem+'.pt'),snapshots/f'epoch{epoch:02d}.pt')
                    volume.commit()
            code=child.wait()
        volume.commit()
        if code:
            print((root/name).read_text()[-9000:],flush=True)
            raise RuntimeError(f'{name}: exit {code}')
    execute([sys.executable,'-P','-u','/opt/iterate/test_gdn_iteration_gpu.py'],'gpu-validation.log')
    execute([sys.executable,'-P','-u','-m','zoology.launch','/opt/iterate/zoo_gdn_iteration_configs.py'],'train.log')
    decision=json.loads((root/'decision.json').read_text())
    status=json.loads((root/(stem+'.json')).read_text())
    result=dict(**decision,complete=status['complete'],trial=trial)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    print('RESULT '+json.dumps(result),flush=True)
    return result

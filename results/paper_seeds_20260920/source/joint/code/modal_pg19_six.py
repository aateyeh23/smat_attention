"""Six matched PG19 runs: plain GDN/Mamba-2 and SMAT d=2/3 on each backbone."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import modal

LOCAL = Path(__file__).resolve().parent
FILES = ('smat_write_hash_triton.py', 'lm/test_write_hash_fused_gpu.py',
    'content_addr.py', 'smat_read_triton.py', 'smat_write_hash_grad.py',
    'smat_address_reads.py', 'smat_gdn_transport.py', 'zoo_gdn_transport.py',
    'zoo_smat_gdn.py', 'zoo_smat_mixer.py', 'lm/gdn_smat_scale.py',
    'lm/gdn_smat_transport_scale.py', 'lm/mamba_smat_scale.py',
    'lm/train_pg19_scale.py', 'lm/test_pg19_rank64_gpu.py',
    'lm/run_pg19_rank64.py', 'lm/pg19_baselines.py', 'lm/test_pg19_baselines_gpu.py')
OPT_FILES = ('smat_read_tiled.py','smat_write_hash_tiled.py','smat_gdn_tiled.py',
             'pg19_optimized_kernels.py','lm/run_pg19_optimized.py')
app = modal.App('pg19-six-d23')
data = modal.Volume.from_name('smat-pg19-scale-data-v1')
volume = modal.Volume.from_name('pg19-w768-rank64')
kernel_cache = modal.Volume.from_name('pg19-transport-optimization')
if modal.is_local():
    from modal_mqar_four_reads import image
    for name in FILES + OPT_FILES:
        image = image.add_local_file(LOCAL/name, '/opt/pg19-rank64/'+name)
    image=image.add_local_file(LOCAL/'results/pg19_transport_opt/summary.json',
                               '/opt/pg19-optimization-validation.json')
else:
    image = modal.Image.debian_slim()


def arm_name(family, d):
    return family + ('-baseline' if d == 1 else f'-smat-d{d}')


@app.function(image=image, gpu='H100', cpu=4, memory=49152, timeout=43200,
              max_containers=6, volumes={'/data':data, '/checkpoints':volume,'/kernel-cache':kernel_cache})
def run(family: str, d: int, tokens: int = 500_000_000,
        campaign: str = 'six-500m-20260917', optimized: bool = False):
    if family not in ('gdn', 'mamba2') or d not in (1,2,3):
        raise ValueError('Only plain baselines and SMAT d=2/3 are authorized')
    if not campaign or Path(campaign).name != campaign:
        raise ValueError('Campaign must be one directory name')
    if optimized and (family!='gdn' or d==1):
        raise ValueError('Optimized kernels are only for GDN + SMAT')
    files=FILES+(OPT_FILES if optimized else ())
    name = arm_name(family, d)
    batch = 1 if family == 'mamba2' and d == 3 else 2
    heads, head_dim = (6,128) if family == 'gdn' else (24,64)
    variant = (family+'_baseline' if d == 1 else
               'updated_transport' if family == 'gdn' else 'mamba2_fixed_write')
    data.reload()
    volume.reload()
    root = Path('/checkpoints/seed123')/campaign/name
    root.mkdir(parents=True, exist_ok=True)
    sources = {n:hashlib.sha256(Path('/opt/pg19-rank64', n).read_bytes()).hexdigest()
               for n in files}
    manifest = dict(campaign=campaign, family=family, d=d, baseline=d==1,
        width=768, layers=16, ffn_width=2048, heads=heads, head_dim=head_dim,
        read_rank=None if d==1 else 64, length=16384, seed=123,
        target_tokens=tokens, max_hours=11.8, lr=.0003, warmup_tokens=40_000_000,
        batch=batch, global_batch_rows=8, activation_checkpointing=False,
        full_sequence_recurrence=d==1, source_sha256=sources)
    if optimized:manifest['kernel_optimization']='tiled_fp32_v1'
    path = root/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Changed campaign recipe or source; refusing to overwrite')
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    for source in files:
        dst = root/'sources'/source
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path('/opt/pg19-rank64', source), dst)
    (root/'runtime-pip-freeze.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    env = dict(os.environ,
        PYTHONPATH='/opt/pg19-rank64/lm:/opt/pg19-rank64:'+os.environ['PYTHONPATH'],
        PYTHONSAFEPATH='1', PG19_MEMORY_VARIANT=variant,
        PG19_EXPECTED_SOURCES=json.dumps(sources), TRITON_CACHE_DIR='/tmp/pg19-triton-cache',
        SMAT_WRITE_HASH_BACKEND='triton')
    if optimized and Path('/kernel-cache/triton-cache.tar.gz').exists():
        Path(env['TRITON_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
        with tarfile.open('/kernel-cache/triton-cache.tar.gz') as archive:
            archive.extractall(env['TRITON_CACHE_DIR'],filter='data')

    def phase(state, **details):
        (root/'run_state.json').write_text(json.dumps(
            dict(state=state, updated=time.time(), arm=name, **details), indent=2)+'\n')
        volume.commit()
        print('PHASE '+json.dumps(dict(arm=name, state=state, **details)), flush=True)

    def execute(args, filename):
        last = time.monotonic()
        with (root/filename).open('a') as fp:
            child = subprocess.Popen([sys.executable, '-P', '-u',
                '/opt/pg19-rank64/lm/'+('run_pg19_optimized.py' if optimized else 'run_pg19_rank64.py'), *args], env=env,
                cwd='/opt/pg19-rank64', stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True)
            for line in child.stdout:
                fp.write(line)
                fp.flush()
                print(name+' '+line, end='', flush=True)
                if '"event": "checkpoint"' in line or time.monotonic()-last > 45:
                    volume.commit()
                    last = time.monotonic()
            code = child.wait()
        volume.commit()
        if code:
            raise RuntimeError(f'{name}: {filename} exited with code {code}')

    try:
        phase('validating')
        execute(['validate', '--d', str(d)], 'validation.log')
        (root/'validated.json').write_text(json.dumps(dict(passed=True, sources=sources)))
        args = ['train', '--d', str(d), '--data', '/data/pg19-16k-2b',
            '--output', str(root), '--tokens', str(tokens), '--width', '768',
            '--layers', '16', '--ffn-width', '2048', '--heads', str(heads),
            '--head-dim', str(head_dim), '--batch', str(batch), '--global-batch-rows', '8',
            '--max-hours', '11.8']
        if d != 1:
            args += ['--read-rank', '64']
        status_path = root/'status.json'
        saved = json.loads(status_path.read_text()) if status_path.exists() else {}
        if saved.get('step', 0) < 4:
            phase('pilot')
            execute([*args, '--max-steps', '4', '--log-every', '1'], 'train.log')
            saved = json.loads(status_path.read_text())
            if saved['step'] != 4 or not (root/'latest.pt').exists():
                raise RuntimeError('Full-size pilot did not reach its checkpoint')
        phase('training', resumed_step=saved['step'])
        execute([*args, '--log-every', '10'], 'train.log')
        status = json.loads(status_path.read_text())
        phase('complete' if status['complete'] else 'paused', tokens=status['tokens'])
        return dict(arm=name, **status)
    except Exception as error:
        phase('failed', error=str(error))
        raise


@app.function(timeout=46800)
def sweep(tokens: int = 500_000_000, campaign: str = 'six-500m-20260917'):
    calls = {arm_name(f,d):run.spawn(f,d,tokens,campaign)
             for f in ('gdn','mamba2') for d in (1,2,3)}
    print('CALLS '+json.dumps({k:c.object_id for k,c in calls.items()}), flush=True)
    results = {}
    for name, call in calls.items():
        try:
            results[name] = call.get()
        except Exception as error:
            results[name] = dict(error=str(error))
    print('RESULTS '+json.dumps(results), flush=True)
    return results


@app.function(image=image,cpu=4,memory=16384,timeout=46800,
              volumes={'/checkpoints':volume})
def resume_gdn_optimized(source_campaign: str='six-500m-20260917',
                         campaign: str='six-500m-20260917-opt-v1'):
    """Copy paused production checkpoints, verify provenance, and resume only GDN SMAT."""
    import torch
    for value in (source_campaign,campaign):
        if not value or Path(value).name!=value:raise ValueError('Invalid campaign name')
    if source_campaign==campaign:raise ValueError('Retain original campaign as the immutable parent')
    volume.reload()
    base_hashes={n:hashlib.sha256(Path('/opt/pg19-rank64',n).read_bytes()).hexdigest() for n in FILES}
    opt_hashes={n:hashlib.sha256(Path('/opt/pg19-rank64',n).read_bytes()).hexdigest() for n in OPT_FILES}
    proof=json.loads(Path('/opt/pg19-optimization-validation.json').read_text())
    if proof['current_implementation_sha256']!=opt_hashes:
        raise ValueError('Optimized sources differ from the validated implementation')
    planned=[]
    # Check both arms before any checkpoint copies or launches.
    for d in (2,3):
        name=arm_name('gdn',d)
        source=Path('/checkpoints/seed123')/source_campaign/name
        root=Path('/checkpoints/seed123')/campaign/name
        if root.exists():raise ValueError(f'Destination already exists: {root}; refusing duplicate launch')
        manifest=json.loads((source/'manifest.json').read_text())
        state=json.loads((source/'run_state.json').read_text())
        status=json.loads((source/'status.json').read_text())
        if state['state']!='paused' or status['complete']:
            raise ValueError(f'{name} is not a paused incomplete run')
        if manifest['source_sha256']!=base_hashes:
            raise ValueError(f'{name}: original model or trainer sources changed')
        if manifest['family']!='gdn' or manifest['d']!=d or manifest['target_tokens']!=500_000_000:
            raise ValueError('Unexpected source experiment')
        if (source/'continuation.json').exists():
            raise ValueError(f'{name} already has a continuation')
        if proof['runs'][str(d)]['pilot']['resumed_step']!=status['step']:
            raise ValueError('Paused checkpoint is not the checkpoint used in validation')
        planned.append((d,name,source,root,manifest,status))
    for d,name,source,root,manifest,status in planned:
        root.mkdir(parents=True)
        temporary=root/'latest.copying'
        digest=hashlib.sha256()
        with (source/'latest.pt').open('rb') as src, temporary.open('wb') as dst:
            while block:=src.read(8*1024*1024):
                digest.update(block);dst.write(block)
        expected_digest=digest.hexdigest()
        with temporary.open('rb') as copied:
            actual=hashlib.file_digest(copied,'sha256').hexdigest()
        if actual!=expected_digest:raise RuntimeError('Checkpoint copy checksum mismatch')
        temporary.replace(root/'latest.pt')
        for filename in ('recipe.json','status.json','metrics.jsonl'):
            shutil.copy2(source/filename,root/filename)
        saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
        recipe=json.loads((root/'recipe.json').read_text())
        if saved['recipe']!=recipe or saved['step']!=status['step'] or saved['cursor']*16384!=status['tokens']:
            raise ValueError('Checkpoint state and recipe do not match the saved status')
        for key in ('model','optimizer','rng_cpu','rng_cuda','next_eval','next_milestone'):
            if key not in saved:raise ValueError(f'Missing resumable state: {key}')
        del saved
        migration=dict(optimization='tiled_fp32_v1',source_campaign=source_campaign,
            campaign=campaign,arm=name,resumed_step=status['step'],tokens=status['tokens'],
            source_checkpoint=str((source/'latest.pt').relative_to('/checkpoints')),
            checkpoint_sha256=expected_digest,source_manifest=manifest,
            optimized_source_sha256=opt_hashes,validation_app=proof['app'],created=time.time())
        (root/'migration.json').write_text(json.dumps(migration,indent=2)+'\n')
        (root/'run_state.json').write_text(json.dumps(dict(state='queued',arm=name,
            updated=time.time(),resumed_step=status['step'],tokens=status['tokens'])))
        (source/'continuation.json').write_text(json.dumps(dict(campaign=campaign,arm=name,
            optimization='tiled_fp32_v1',checkpoint_sha256=expected_digest,created=time.time()),indent=2)+'\n')
        volume.commit()
        print('MIGRATED '+json.dumps(migration),flush=True)
    calls={name:run.spawn('gdn',d,500_000_000,campaign,True) for d,name,*_ in planned}
    print('CALLS '+json.dumps({n:c.object_id for n,c in calls.items()}),flush=True)
    results={}
    for name,call in calls.items():
        try:results[name]=call.get()
        except Exception as error:results[name]=dict(error=str(error))
    print('RESULTS '+json.dumps(results),flush=True)
    return results

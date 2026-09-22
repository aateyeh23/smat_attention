"""Continue all six PG19 models from 750M to 1B with automatic handoff."""
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
SOURCE_CAMPAIGN = 'six-750m-constant3e-5-20260917'
CAMPAIGN = 'six-1b-constant3e-5-20260917'
app = modal.App('pg19-six-1b-constant-lr')
data = modal.Volume.from_name('smat-pg19-scale-data-v1')
volume = modal.Volume.from_name('pg19-w768-rank64')
kernel_cache = modal.Volume.from_name('pg19-transport-optimization')
if modal.is_local():
    from modal_pg19_six import image
    image = image.add_local_file(LOCAL/'lm/continue_pg19_scale.py',
                                '/opt/pg19-rank64/lm/continue_pg19_scale.py')
    image = image.add_local_file(LOCAL/'lm/continue_pg19_1b.py',
                                '/opt/pg19-rank64/lm/continue_pg19_1b.py')
else:
    image = modal.Image.debian_slim()


def arm_name(family, d):
    return family + ('-baseline' if d == 1 else f'-smat-d{d}')


@app.function(image=image, gpu='H100', cpu=4, memory=49152, timeout=43200,
              max_containers=6, volumes={'/data':data, '/checkpoints':volume,
                                        '/kernel-cache':kernel_cache})
def run(family: str, d: int, campaign: str=CAMPAIGN):
    if family not in ('gdn', 'mamba2') or d not in (1, 2, 3):
        raise ValueError('Unexpected model')
    if not campaign or Path(campaign).name != campaign:
        raise ValueError('Invalid campaign')
    data.reload(); volume.reload(); kernel_cache.reload()
    name = arm_name(family, d)
    optimized = family == 'gdn' and d > 1
    source_campaign = SOURCE_CAMPAIGN
    source = Path('/checkpoints/seed123')/source_campaign/name
    root = Path('/checkpoints/seed123')/campaign/name
    parent = json.loads((source/'manifest.json').read_text())
    status = json.loads((source/'status.json').read_text())
    if not status['complete'] or status['tokens'] != 750010368:
        raise ValueError('Parent must be the completed 750M run')
    if root.exists():
        raise ValueError(f'Destination exists; refusing duplicate launch: {root}')
    for filename, expected in parent['source_sha256'].items():
        actual = hashlib.sha256(Path('/opt/pg19-rank64', filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Frozen training source changed: {filename}')
    checkpoint = source/'latest.pt'
    with checkpoint.open('rb') as fp:
        checkpoint_hash = hashlib.file_digest(fp, 'sha256').hexdigest()
    trainer = 'lm/continue_pg19_1b.py'
    sources = dict(parent['source_sha256'])
    sources[trainer] = hashlib.sha256(Path('/opt/pg19-rank64', trainer).read_bytes()).hexdigest()
    manifest = dict(parent, campaign=campaign, target_tokens=1000000000,
        lr=3e-5, warmup_tokens=0, lr_schedule='constant', max_hours=11.8,
        source_sha256=sources, parent_campaign=source_campaign,
        parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=checkpoint_hash,
        starting_tokens=status['tokens'], starting_step=status['step'],
        parent_elapsed_seconds=status['elapsed_seconds'],
        resume_optimizer=True, resume_rng=True, resume_data_cursor=True)
    root.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for filename in sources:
        dst = root/'sources'/filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path('/opt/pg19-rank64', filename), dst)
    (root/'runtime-pip-freeze.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    variant = (family+'_baseline' if d == 1 else
               'updated_transport' if family == 'gdn' else 'mamba2_fixed_write')
    env = dict(os.environ, PYTHONPATH='/opt/pg19-rank64/lm:/opt/pg19-rank64:'+os.environ['PYTHONPATH'],
        PYTHONSAFEPATH='1', PG19_MEMORY_VARIANT=variant, PG19_OPTIMIZED=str(int(optimized)),
        SMAT_WRITE_HASH_BACKEND='triton', TRITON_CACHE_DIR='/tmp/pg19-continue-triton')
    archive_path = Path('/kernel-cache/triton-cache.tar.gz')
    if optimized and archive_path.exists():
        Path(env['TRITON_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(env['TRITON_CACHE_DIR'], filter='data')

    def phase(state, **extra):
        record = dict(state=state, arm=name, updated=time.time(), **extra)
        (root/'run_state.json').write_text(json.dumps(record, indent=2)+'\n')
        volume.commit()
        print('PHASE '+json.dumps(record), flush=True)

    args = ['--d', str(d), '--data', '/data/pg19-16k-2b', '--output', str(root),
        '--resume-from', str(checkpoint), '--tokens', '1000000000', '--lr', '3e-5',
        '--warmup-tokens', '0', '--width', '768', '--layers', '16', '--ffn-width', '2048',
        '--heads', str(parent['heads']), '--head-dim', str(parent['head_dim']),
        '--batch', str(parent['batch']), '--global-batch-rows', '8', '--log-every', '10',
        '--max-hours', '11.8']
    if d > 1:
        args += ['--read-rank', '64']
    try:
        phase('training', resumed_step=status['step'], tokens=status['tokens'], learning_rate=3e-5)
        with (root/'train.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-P', '-u', '/opt/pg19-rank64/'+trainer, *args],
                env=env, cwd='/opt/pg19-rank64', stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True)
            last = time.monotonic()
            for line in child.stdout:
                log.write(line); log.flush()
                print(name+' '+line, end='', flush=True)
                if '"event": "checkpoint"' in line or time.monotonic()-last > 45:
                    volume.commit(); last = time.monotonic()
            code = child.wait()
        volume.commit()
        if code:
            raise RuntimeError(f'Training exited with code {code}; see train.log')
        final = json.loads((root/'status.json').read_text())
        phase('complete' if final['complete'] else 'paused', tokens=final['tokens'])
        return dict(arm=name, **final)
    except Exception as error:
        phase('failed', error=str(error))
        raise


@app.function(timeout=46800, volumes={'/checkpoints':volume})
def sweep(campaign: str=CAMPAIGN):
    """Wait on CPU for each 750M parent, then launch its 1B continuation."""
    pending = {arm_name(f, d):(f, d) for f in ('gdn', 'mamba2') for d in (1, 2, 3)}
    calls = {}
    results = {}
    deadline = time.monotonic()+11.8*3600
    while pending:
        volume.reload()
        for name, (family, d) in list(pending.items()):
            source = Path('/checkpoints/seed123')/SOURCE_CAMPAIGN/name
            state = json.loads((source/'run_state.json').read_text())
            if state['state'] in ('failed', 'paused'):
                results[name] = dict(error='750M parent did not complete', parent_state=state)
                del pending[name]
                print('PARENT_ERROR '+json.dumps(dict(arm=name, **results[name])), flush=True)
                continue
            if state['state'] != 'complete':
                continue
            status = json.loads((source/'status.json').read_text())
            if not status['complete'] or status['tokens'] != 750010368:
                raise ValueError(f'{name}: inconsistent completed parent')
            calls[name] = run.spawn(family, d, campaign)
            del pending[name]
            print('LAUNCHED '+json.dumps(dict(arm=name, call_id=calls[name].object_id,
                parent_tokens=status['tokens'], target_tokens=1000000000)), flush=True)
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for 750M parents')
            print('WAITING '+json.dumps(list(pending)), flush=True)
            time.sleep(30)
    for arm, call in calls.items():
        try:
            results[arm] = call.get()
        except Exception as error:
            results[arm] = dict(error=str(error))
    print('RESULTS '+json.dumps(results), flush=True)
    return results

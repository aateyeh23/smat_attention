"""Resume only the two paused SMAT d3 runs to their original 750M target."""
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
CAMPAIGN = 'six-750m-constant3e-5-20260917'
app = modal.App('pg19-d3-finish-750m')
data = modal.Volume.from_name('smat-pg19-scale-data-v1')
volume = modal.Volume.from_name('pg19-w768-rank64')
kernel_cache = modal.Volume.from_name('pg19-transport-optimization')
if modal.is_local():
    from modal_pg19_six import image
    image = image.add_local_file(LOCAL/'lm/continue_pg19_scale.py',
                                '/opt/pg19-rank64/lm/continue_pg19_scale.py')
else:
    image = modal.Image.debian_slim()


def arm_name(family, d):
    return family + ('-baseline' if d == 1 else f'-smat-d{d}')


@app.function(image=image, gpu='H100', cpu=4, memory=49152, timeout=43200,
              max_containers=2, volumes={'/data':data, '/checkpoints':volume,
                                        '/kernel-cache':kernel_cache})
def run(family: str, d: int, campaign: str=CAMPAIGN):
    if family not in ('gdn', 'mamba2') or d != 3:
        raise ValueError('Unexpected model')
    if not campaign or Path(campaign).name != campaign:
        raise ValueError('Invalid campaign')
    data.reload(); volume.reload(); kernel_cache.reload()
    name = arm_name(family, d)
    optimized = family == 'gdn' and d > 1
    root = Path('/checkpoints/seed123')/campaign/name
    parent = json.loads((root/'manifest.json').read_text())
    status = json.loads((root/'status.json').read_text())
    state = json.loads((root/'run_state.json').read_text())
    if state['state'] != 'paused' or status['complete']:
        raise ValueError('Only a paused incomplete run can be resumed')
    if parent['target_tokens'] != 750000000 or parent['lr'] != 3e-5 or parent['lr_schedule'] != 'constant':
        raise ValueError('Unexpected continuation recipe')
    if not (root/'latest.pt').exists():
        raise FileNotFoundError(root/'latest.pt')
    for filename, expected in parent['source_sha256'].items():
        if hashlib.sha256(Path('/opt/pg19-rank64',filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen training source changed: '+filename)
    checkpoint = Path(parent['parent_checkpoint'])
    trainer = 'lm/continue_pg19_scale.py'
    # Keep the original recipe's parent path; trainer resumes root/latest.pt first.
    (root/'STOP').unlink(missing_ok=True)
    (root/'resume-750m.json').write_text(json.dumps(dict(request='Finish d3 to 750M only',
        resumed_tokens=status['tokens'], resumed_step=status['step'], created=time.time()),indent=2)+'\n')
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
        '--resume-from', str(checkpoint), '--tokens', '750000000', '--lr', '3e-5',
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


@app.function(timeout=46800)
def sweep(campaign: str=CAMPAIGN):
    calls = {arm_name(f, d):run.spawn(f, d, campaign)
             for f in ('gdn', 'mamba2') for d in (3,)}
    print('CALLS '+json.dumps({arm:call.object_id for arm, call in calls.items()}), flush=True)
    results = {}
    for arm, call in calls.items():
        try:
            results[arm] = call.get()
        except Exception as error:
            results[arm] = dict(error=str(error))
    print('RESULTS '+json.dumps(results), flush=True)
    return results

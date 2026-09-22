"""Read-only MK-NIAH-1 evaluation of the six 500M-token PG19 checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import modal

LOCAL = Path(__file__).resolve().parent
DATA = LOCAL/'results/pg19_ruler_mk_niah1'
ARMS = [f'{family}-{suffix}' for family in ('gdn','mamba2')
        for suffix in ('baseline','smat-d2','smat-d3')]
app = modal.App('pg19-ruler-recurrent')
checkpoints = modal.Volume.from_name('pg19-w768-rank64')
kernel_cache = modal.Volume.from_name('pg19-transport-optimization')
results = modal.Volume.from_name('pg19-ruler-evaluation', create_if_missing=True)
if modal.is_local():
    from modal_pg19_six import image
    for name in ('eval_pg19_ruler_recurrent.py', 'pg19_recurrent.py'):
        image = image.add_local_file(LOCAL/'lm'/name, '/opt/pg19-rank64/lm/'+name)
    for filename in ('encoded.jsonl', 'dataset-manifest.json', 'ruler-metrics.py'):
        image = image.add_local_file(DATA/filename, '/ruler-data/'+filename)
else:
    image = modal.Image.debian_slim()


@app.function(image=image, gpu='H100', cpu=4, memory=49152, timeout=43200,
              max_containers=6, volumes={'/checkpoints':checkpoints,
                                        '/kernel-cache':kernel_cache, '/results':results})
def evaluate(arm: str, limit: int=0, batch: int=4):
    if arm not in ARMS:
        raise ValueError(arm)
    checkpoints.reload()
    kernel_cache.reload()
    results.reload()
    campaign = 'six-500m-20260917-opt-v1' if arm.startswith('gdn-smat') else 'six-500m-20260917'
    checkpoint = Path('/checkpoints/seed123')/campaign/arm/'model-tokens-0500006912.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output = Path('/results/mk-niah1-16k-500m')/('recurrent' if not limit else f'recurrent-pilot-{limit}')/arm
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='/opt/pg19-rank64/lm:/opt/pg19-rank64:'+os.environ['PYTHONPATH'],
               PYTHONSAFEPATH='1', SMAT_WRITE_HASH_BACKEND='triton',
               TRITON_CACHE_DIR='/tmp/pg19-ruler-triton')
    archive_path = Path('/kernel-cache/triton-cache.tar.gz')
    if archive_path.exists():
        Path(env['TRITON_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(env['TRITON_CACHE_DIR'], filter='data')
    def phase(state, **extra):
        (output/'run_state.json').write_text(json.dumps(dict(state=state, arm=arm, updated=time.time(), **extra)))
        results.commit()
    try:
        phase('evaluating')
        while True:
            retry = output/'retry-batch.json'
            retry.unlink(missing_ok=True)
            command = [sys.executable, '-P', '-u', '/opt/pg19-rank64/lm/eval_pg19_ruler_recurrent.py',
                '--checkpoint', str(checkpoint), '--data', '/ruler-data', '--output', str(output),
                '--limit', str(limit), '--batch', str(batch)]
            with (output/'eval.log').open('a') as log:
                child = subprocess.Popen(command, cwd='/opt/pg19-rank64', env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                last_commit = time.monotonic()
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(arm+' '+line, end='', flush=True)
                    if line.startswith('PROGRESS ') or time.monotonic()-last_commit > 45:
                        results.commit()
                        last_commit = time.monotonic()
                code = child.wait()
            results.commit()
            if code == 0:
                break
            if retry.exists():
                batch = json.loads(retry.read_text())['batch']
                print(f'{arm}: retrying with batch {batch}', flush=True)
                continue
            raise RuntimeError(f'Evaluation exit {code}; see eval.log')
        phase('complete')
        return json.loads((output/'result.json').read_text())
    except Exception as error:
        phase('failed', error=str(error))
        raise


@app.function(timeout=46800)
def sweep(arms: str=','.join(ARMS), limit: int=0, batch: int=4):
    names = arms.split(',')
    if len(set(names)) != len(names) or any(name not in ARMS for name in names):
        raise ValueError('Invalid or duplicate arms')
    calls = {arm:evaluate.spawn(arm, limit, batch) for arm in names}
    print('CALLS '+json.dumps({arm:call.object_id for arm,call in calls.items()}), flush=True)
    summary = {}
    for arm, call in calls.items():
        try:
            summary[arm] = call.get()
        except Exception as error:
            summary[arm] = dict(error=str(error))
    print('RESULTS '+json.dumps(summary), flush=True)
    return summary

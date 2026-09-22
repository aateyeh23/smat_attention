"""Verify and run n-gram speculative greedy decoding for the PG19 RULER eval."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import modal

LOCAL = Path(__file__).resolve().parent
ARMS = [f'{family}-{suffix}' for family in ('gdn','mamba2')
        for suffix in ('baseline','smat-d2','smat-d3')]
app = modal.App('pg19-recurrent-validation')
checkpoints = modal.Volume.from_name('pg19-w768-rank64')
cache = modal.Volume.from_name('pg19-transport-optimization')
results = modal.Volume.from_name('pg19-ruler-evaluation')
if modal.is_local():
    from modal_pg19_ruler import image
    for name in ('pg19_recurrent.py','test_pg19_recurrent_gpu.py'):
        image = image.add_local_file(LOCAL/'lm'/name, '/opt/pg19-rank64/lm/'+name)
else:
    image = modal.Image.debian_slim()


@app.function(image=image, gpu='H100', cpu=4, memory=49152, timeout=43200,
              max_containers=6, volumes={'/checkpoints':checkpoints,'/kernel-cache':cache,'/results':results})
def evaluate(arm: str='mamba2-smat-d3', verify_only: bool=True, batch: int=4):
    assert arm in ARMS
    checkpoints.reload()
    cache.reload()
    results.reload()
    campaign = 'six-500m-20260917-opt-v1' if arm.startswith('gdn-smat') else 'six-500m-20260917'
    checkpoint = Path('/checkpoints/seed123')/campaign/arm/'model-tokens-0500006912.pt'
    base = Path('/results/mk-niah1-16k-500m')
    output = base/'recurrent-validation'/arm
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='/opt/pg19-rank64/lm:/opt/pg19-rank64:'+os.environ['PYTHONPATH'],
               PYTHONSAFEPATH='1', SMAT_WRITE_HASH_BACKEND='triton', TRITON_CACHE_DIR='/tmp/ruler-fast')
    archive_path = Path('/kernel-cache/triton-cache.tar.gz')
    if archive_path.exists():
        Path(env['TRITON_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(env['TRITON_CACHE_DIR'], filter='data')
    def phase(state, **extra):
        (output/'run_state.json').write_text(json.dumps(dict(state=state,arm=arm,updated=time.time(),**extra)))
        results.commit()
    phase('verifying' if verify_only else 'evaluating')
    command = [sys.executable,'-P','-u','/opt/pg19-rank64/lm/test_pg19_recurrent_gpu.py',
        '--checkpoint',str(checkpoint),'--data','/ruler-data/encoded.jsonl',
        '--output',str(output/'verification.json')]
    try:
        with (output/'eval.log').open('a') as log:
            child = subprocess.Popen(command, cwd='/opt/pg19-rank64', env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            last = time.monotonic()
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(arm+' '+line,end='',flush=True)
                if line.startswith('PROGRESS ') or time.monotonic()-last > 45:
                    results.commit()
                    last=time.monotonic()
            code=child.wait()
        results.commit()
        if code:
            raise RuntimeError(f'Exit {code}')
        phase('complete')
        return json.loads((output/'verification.json').read_text())
    except Exception as error:
        phase('failed',error=str(error))
        raise


@app.function(timeout=46800)
def sweep(verify_only: bool=True, arms: str=','.join(ARMS)):
    names=arms.split(',')
    assert len(set(names))==len(names) and all(name in ARMS for name in names)
    calls = {arm:evaluate.spawn(arm,verify_only) for arm in names}
    print('CALLS '+json.dumps({arm:c.object_id for arm,c in calls.items()}),flush=True)
    summary={}
    for arm,call in calls.items():
        try:
            summary[arm]=call.get()
        except Exception as error:
            summary[arm]=dict(error=str(error))
    print('RESULTS '+json.dumps(summary),flush=True)
    return summary

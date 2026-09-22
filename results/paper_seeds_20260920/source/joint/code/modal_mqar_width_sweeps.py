"""Expanded MQAR widths, frozen original routing backend, epoch checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

LOCAL = Path(__file__).resolve().parent
app = modal.App('smat-mqar-four-reads-width-sweeps')
volume = modal.Volume.from_name('smat-mqar-width-sweeps', create_if_missing=True)
if modal.is_local():
    from modal_mqar_four_reads import image as cached_image
    image = cached_image
    for filename in ('zoo_mamba_four_reads.py', 'zoo_mqar_four_reads_sweep_configs.py'):
        image = image.add_local_file(LOCAL/filename, '/opt/smat/'+filename)
else:
    image = modal.Image.debian_slim()


@app.function(image=image, gpu='H100', cpu=4, memory=32768,
              volumes={'/sweep-results': volume}, timeout=10800, max_containers=6)
def train(family: str, width: int, d: int, smoke: bool = False):
    volume.reload()
    root = Path('/sweep-results/seed123') / ('smoke' if smoke else 'full') / family
    root.mkdir(parents=True, exist_ok=True)
    name = f'w{width}-d{d}'
    status = root / (name+'.json')
    log = root / (name+'.log')
    env = dict(os.environ, MQAR_FAMILY=family, ZOO_DM=str(width), ZOO_DS=str(d),
               ZOO_EPOCHS='2' if smoke else '32', MQAR_RUN_DIR=str(root),
               MQAR_MAX_MINUTES='165', TRITON_CACHE_DIR=f'/tmp/triton-{family}-{width}-{d}')
    if smoke:
        env['ZOO_SMOKE'] = '1'
    else:
        env.pop('ZOO_SMOKE', None)
    previous = None
    print(json.dumps(dict(event='start', family=family, width=width, d=d, smoke=smoke)), flush=True)
    with log.open('a') as fp:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'zoology.launch',
                                  '/opt/smat/zoo_mqar_four_reads_sweep_configs.py'],
                                 cwd='/opt/smat/the GPU cluster', env=env, stdout=fp, stderr=subprocess.STDOUT)
        while child.poll() is None:
            if status.exists():
                current = status.read_text()
                if current != previous:
                    try:
                        payload = json.loads(current)
                    except json.JSONDecodeError:
                        time.sleep(1)
                        continue
                    previous = current
                    volume.commit()
                    print(json.dumps(dict(family=family, width=width, d=d, smoke=smoke, **payload)), flush=True)
            time.sleep(5)
        code = child.returncode
    volume.commit()
    if code:
        print(log.read_text()[-12000:], flush=True)
        raise RuntimeError(f'{family} w{width} d{d} failed: exit {code}')
    result = dict(family=family, width=width, d=d, smoke=smoke, **json.loads(status.read_text()))
    if not result['complete']:
        raise RuntimeError('Time limit reached; saved checkpoint is available for resume')
    print('FINISHED ' + json.dumps(result), flush=True)
    return result


@app.function(timeout=43200)
def sweep(family: str = 'gdn'):
    if family not in ('gdn', 'mamba2'):
        raise ValueError(family)
    # A real trainer smoke checks the newly configured maximum width and route count.
    print('SMOKE ' + json.dumps(train.remote(family, 64, 4, True)), flush=True)
    widths = (32, 64) if family == 'gdn' else (16, 32, 64)
    calls = {(w, d): train.spawn(family, w, d) for w in widths for d in (1, 2, 3, 4)}
    print('CALLS ' + json.dumps({f'w{w}-d{d}': c.object_id for (w,d),c in calls.items()}), flush=True)
    results = [call.get() for call in calls.values()]
    print('FINAL_RESULTS ' + json.dumps(results), flush=True)
    return json.dumps(results)

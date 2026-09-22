"""Matched 2B-token PG19 experiment; launch only after fused GPU validation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import modal

LOCAL = Path(__file__).resolve().parent
app = modal.App('smat-pg19-2b-pretrain')
data = modal.Volume.from_name('smat-pg19-scale-data-v1')
checkpoints = modal.Volume.from_name('smat-pg19-2b-checkpoints', create_if_missing=True)
if modal.is_local():
    from modal_mqar_four_reads import image as cached_image
    image = cached_image
    for name in ('content_addr.py', 'smat_read_triton.py', 'lm/gdn_smat_scale.py',
                 'lm/train_pg19_scale.py', 'lm/test_scale_lm_gpu.py'):
        image = image.add_local_file(LOCAL/name, '/opt/smat-optimized/'+name)
else:
    image = modal.Image.debian_slim()


def command_run(command, output, commit_every=60):
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='/opt/smat-optimized:'+os.environ['PYTHONPATH'])
    last_commit = time.monotonic()
    with (output/'train.log').open('a') as fp:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, cwd='/opt/smat-optimized', env=env)
        for line in child.stdout:
            fp.write(line)
            fp.flush()
            print(line, end='', flush=True)
            if '"event": "checkpoint"' in line or time.monotonic()-last_commit > commit_every:
                checkpoints.commit()
                last_commit = time.monotonic()
        code = child.wait()
    checkpoints.commit()
    if code:
        raise RuntimeError(f'Training process exited {code}; see {output}/train.log')


@app.function(image=image, gpu='H100', cpu=4, memory=32768, timeout=1800,
              volumes={'/data': data, '/checkpoints': checkpoints})
def validate():
    data.reload()
    command_run([sys.executable, '-u', '/opt/smat-optimized/lm/test_scale_lm_gpu.py',
                 '--data', '/data/pg19-16k-1b'], Path('/checkpoints/gpu-validation'))
    return 'GPU inference and checkpoint checks passed'


@app.function(image=image, gpu='H100', cpu=4, memory=32768, timeout=43200,
              max_containers=4, volumes={'/data': data, '/checkpoints': checkpoints})
def train(d: int):
    data.reload()
    checkpoints.reload()
    root = Path('/checkpoints/seed123')/f'd{d}'
    command_run([sys.executable, '-u', '/opt/smat-optimized/lm/train_pg19_scale.py',
                 '--d', str(d), '--data', '/data/pg19-16k-2b', '--output', str(root),
                 '--batch', '3', '--global-batch-rows', '8'], root)
    return json.loads((root/'status.json').read_text())


@app.function(timeout=46800)
def sweep():
    calls = {d: train.spawn(d) for d in (1,2,3,4)}
    print('CALLS '+json.dumps({d: c.object_id for d,c in calls.items()}), flush=True)
    results = {d: c.get() for d,c in calls.items()}
    print('RESULTS '+json.dumps(results), flush=True)
    return results

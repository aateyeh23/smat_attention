"""GPU kernel validation, then bounded end-to-end PG19 throughput pilots."""
import json
import os
from pathlib import Path
import subprocess
import sys
import modal

LOCAL = Path(__file__).resolve().parent
app = modal.App('smat-fused-kernel-pilot')
volume = modal.Volume.from_name('smat-pg19-scale-pilot', create_if_missing=True)
if modal.is_local():
    from modal_mqar_four_reads import image as cached_image
    image = cached_image
    for name in ('content_addr.py', 'smat_read_triton.py', 'smat_aggregation_experiments.py', 'test_smat_read_gpu.py', 'test_incidence_gpu.py', 'lm/gdn_smat_scale.py'):
        image = image.add_local_file(LOCAL/name, '/opt/smat-optimized/'+name)
    image = image.add_local_file(LOCAL/'results/pg19_scale_pilot/pg19-pilot.bin', '/opt/pg19-pilot.bin')
else:
    image = modal.Image.debian_slim()


@app.function(image=image, gpu='H100', cpu=4, memory=32768,
              volumes={'/pilot-results': volume}, timeout=1800, max_containers=4)
def run(d: int = 0, backend: str = 'triton', fused_norm: bool = False, batch: int = 2):
    name = 'kernel-validation-v4' if d == 0 else f'd{d}-{backend}-norm{int(fused_norm)}-b{batch}-v4'
    output = Path('/pilot-results')/(name+'.json')
    if d == 0:
        cmd = [sys.executable, '-u', '/opt/smat-optimized/test_smat_read_gpu.py']
    else:
        cmd = [sys.executable, '-u', '/opt/smat-optimized/lm/gdn_smat_scale.py',
               '--d', str(d), '--batch', str(batch), '--checkpoint', '0', '--loss-chunk', '4096',
               '--read-backend', backend, '--steps', '20', '--warmup', '5',
               '--fused-norm', str(int(fused_norm)),
               '--data', '/opt/pg19-pilot.bin', '--output', str(output)]
    with output.with_suffix('.log').open('w') as fp:
        env = dict(os.environ, PYTHONPATH='/opt/smat-optimized:'+os.environ['PYTHONPATH'])
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, cwd='/opt/smat-optimized', env=env)
        for line in child.stdout:
            fp.write(line)
            fp.flush()
            print(line, end='', flush=True)
        code = child.wait()
    volume.commit()
    if code:
        raise RuntimeError(f'{name} failed: exit {code}')
    return json.loads(output.read_text()) if d else {'correctness': 'passed'}


@app.function(timeout=3600)
def sweep():
    print('VALIDATION '+json.dumps(run.remote()), flush=True)
    calls = {d: run.spawn(d) for d in (2, 3, 4)}
    print('FUSED_RESULTS '+json.dumps({d:c.get() for d,c in calls.items()}), flush=True)


@app.function(image=image, gpu='H100', cpu=4, memory=32768, timeout=2400,
              volumes={'/pilot-results': volume})
def paired():
    # Each pair uses identical seeds, windows, warmup, steps and the same GPU.
    results = {}
    for d in (4, 2, 3, 1):
        results[d] = {backend: run.local(d, backend, backend == 'triton') for backend in ('torch', 'triton')}
        print('PAIRED_RESULT '+json.dumps({d: results[d]}), flush=True)
    Path('/pilot-results/paired-v4.json').write_text(json.dumps(results, indent=2)+'\n')
    volume.commit()
    return results


@app.function(image=image, gpu='H100', cpu=4, memory=32768, timeout=600,
              volumes={'/pilot-results': volume})
def aggregate():
    env = dict(os.environ, PYTHONPATH='/opt/smat-optimized:'+os.environ['PYTHONPATH'])
    child = subprocess.Popen([sys.executable, '-u', '/opt/smat-optimized/test_incidence_gpu.py'],
                             env=env, cwd='/opt/smat-optimized', stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    with Path('/pilot-results/incidence-bench.log').open('w') as fp:
        for line in child.stdout:
            fp.write(line)
            fp.flush()
            print(line, end='', flush=True)
        child.wait()
    volume.commit()
    if child.returncode:
        raise RuntimeError('Aggregation kernel check failed')

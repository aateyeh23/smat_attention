"""Bounded, evaluation-only diagnostics of an archived iteration checkpoint."""
import os
from pathlib import Path
import subprocess
import sys
import modal

app = modal.App('smat-gdn-iteration-eval-only')
volume = modal.Volume.from_name('smat-gdn-w32-d3-iterations')
if modal.is_local():
    from modal_gdn_transport import image
    image = image.add_local_file(
        Path(__file__).with_name('probe_gdn_iteration_checkpoint.py'),
        '/opt/probe/probe.py')
else:
    image = modal.Image.debian_slim()


@app.function(image=image, gpu='H100', cpu=4, memory=32768, timeout=600,
              volumes={'/iterate-results': volume}, max_containers=1)
def probe(trial: str = '02_isolated_hash', normal_only: bool = False):
    volume.reload()
    root = Path('/iterate-results') / trial
    env = dict(os.environ, PROBE_ROOT=str(root),
               PYTHONPATH='/opt/transport:' + os.environ['PYTHONPATH'],
               TRITON_CACHE_DIR='/iterate-results/triton-cache')
    if normal_only:
        env['PROBE_ONLY_NORMAL'] = '1'
    with (root / 'checkpoint-probe.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-u', '/opt/probe/probe.py'],
                                 env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 cwd='/opt/smat/deltaai')
        for line in child.stdout:
            log.write(line)
            log.flush()
            if 'INTERVENTION ' in line or 'PROBE_RESULT ' in line:
                print(line, end='', flush=True)
        code = child.wait()
    volume.commit()
    if code:
        print((root / 'checkpoint-probe.log').read_text()[-8000:], flush=True)
        raise RuntimeError(f'Evaluation probe failed: {code}')
    return (root / 'checkpoint-probe.json').read_text()

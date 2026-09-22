"""Bounded H100 throughput pilot; does not launch the full pretraining sweep."""
import json
from pathlib import Path
import subprocess
import sys

import modal

LOCAL = Path(__file__).resolve().parent
OUT = LOCAL / 'results/pg19_scale_pilot'
app = modal.App('smat-pg19-scale-pilot')
volume = modal.Volume.from_name('smat-pg19-scale-pilot', create_if_missing=True)
if modal.is_local():
    from modal_mqar_four_reads import image as cached_image
    image = (cached_image
             .add_local_file(LOCAL / 'lm/gdn_smat_scale.py', '/opt/smat/deltaai/lm/gdn_smat_scale.py')
             .add_local_file(OUT / 'pg19-pilot.bin', '/opt/pg19-pilot.bin'))
else:
    image = modal.Image.debian_slim()


@app.function(image=image, gpu='H100', cpu=4, memory=32768,
              volumes={'/pilot-results': volume}, timeout=1200, max_containers=4)
def pilot(d: int, fast: bool = False):
    suffix = '-fast' if fast else ''
    output = Path(f'/pilot-results/d{d}{suffix}.json')
    command = [sys.executable, '-u', '/opt/smat/deltaai/lm/gdn_smat_scale.py',
               '--d', str(d), '--data', '/opt/pg19-pilot.bin', '--output', str(output)]
    if fast:
        command += ['--checkpoint', '0', '--loss-chunk', '4096', '--batch', '2', '--profile']
    log = Path(f'/pilot-results/d{d}{suffix}.log')
    with log.open('w') as fp:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, cwd='/opt/smat/deltaai')
        for line in child.stdout:
            fp.write(line)
            fp.flush()
            print(f'[d={d}] {line}', end='', flush=True)
        code = child.wait()
    volume.commit()
    if code:
        return dict(d=d, error=f'Pilot exited {code}', log=str(log))
    return dict(d=d, **json.loads(output.read_text()))


@app.function(timeout=1500)
def sweep():
    calls = {d: pilot.spawn(d) for d in (1, 2, 3, 4)}
    print('CALLS ' + json.dumps({d: call.object_id for d, call in calls.items()}), flush=True)
    results = {d: call.get() for d, call in calls.items()}
    print('PILOT_RESULTS ' + json.dumps(results), flush=True)
    return json.dumps(results)


@app.function(timeout=1500)
def fast_sweep():
    calls = {d: pilot.spawn(d, True) for d in (1, 2, 3, 4)}
    print('CALLS ' + json.dumps({d: call.object_id for d, call in calls.items()}), flush=True)
    results = {d: call.get() for d, call in calls.items()}
    print('FAST_PILOT_RESULTS ' + json.dumps(results), flush=True)
    return json.dumps(results)

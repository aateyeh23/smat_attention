"""One-GPU sequential, checkpointed worker for the requested three MQAR arms."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ['MQAR_FOUR_READS_DIR'])


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def launch(d, folder, minutes, smoke=False):
    env = os.environ.copy()
    env.pop('ZOO_SMOKE', None)
    env.update(ZOO_DM='16', ZOO_DS=str(d), ZOO_EPOCHS='2' if smoke else '32',
               MQAR_RUN_DIR=str(folder), MQAR_MAX_MINUTES=str(minutes))
    if smoke:
        env['ZOO_SMOKE'] = '1'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'w16-d{d}.log').open('a') as log:
        subprocess.run([sys.executable, '-u', '-m', 'zoology.launch',
                        str(ROOT / 'zoo_gdn_four_reads_configs.py')],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def validate():
    """Share validation between dispatcher workers without duplicate smoke writes."""
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'validation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (RUN / 'validated.json').exists():
            return
        job = os.environ['SLURM_JOB_ID']
        with (RUN / f'preflight-{job}.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(ROOT / 'test_four_reads.py'), '--gpu'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        smoke = RUN / 'smoke'
        launch(4, smoke, -1, smoke=True)
        launch(4, smoke, -1, smoke=True)
        assert read(smoke / 'w16-d4.json')['complete']
        assert 'RESUMED' in (smoke / 'w16-d4.log').read_text()
        write(RUN / 'validated.json', dict(job=job, gpu_checks=True, trainer_resume=True))


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 100 * 60
        job = os.environ['SLURM_JOB_ID']
        write(RUN / 'status.json', dict(job=job, phase='preflight', complete=False))
        validate()
        for d in (2, 3, 4):
            if read(RUN / f'w16-d{d}.json').get('complete'):
                continue
            minutes = (deadline - time.monotonic()) / 60 - 5
            if minutes < 5:
                break
            previous = read(RUN / f'w16-d{d}.json').get('next_epoch', 0)
            write(RUN / 'status.json', dict(job=job, phase='training', d=d, complete=False))
            print(f'START d={d} width=16 heads=1 head/state=16 writes=1 reads=4', flush=True)
            launch(d, RUN, minutes)
            current = read(RUN / f'w16-d{d}.json')
            if current.get('next_epoch', 0) <= previous:
                raise RuntimeError(f'd={d} training did not produce a new checkpoint')
            print(f'CHECKPOINT d={d}: {current}', flush=True)
            if not current.get('complete'):
                break
        rows = {str(d): read(RUN / f'w16-d{d}.json') for d in (2, 3, 4)}
        complete = all(row.get('complete') for row in rows.values())
        write(RUN / 'status.json', dict(job=job, phase='complete' if complete else 'continuing',
                                        complete=complete, runs=rows))
        if complete:
            (RUN / 'COMPLETE').write_text('All three MQAR runs completed.\n')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(RUN / 'error.json', dict(job=os.environ.get('SLURM_JOB_ID'),
                                      traceback=traceback.format_exc()))
        raise

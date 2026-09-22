"""Launch the frozen width-16 four-read MQAR sweep on three Modal H100s.

modal run --detach modal_mqar_four_reads.py::sweep
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

LOCAL = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get('MQAR_MODAL_BUNDLE', LOCAL / 'results/mqar_gdn_four_reads/modal_bundle'))
REMOTE = Path('/opt/smat/the GPU cluster')
RUN = Path('/results/four_reads_w16_s123')
app = modal.App('smat-mqar-w16-four-reads')
volume = modal.Volume.from_name('smat-mqar-four-reads-results', create_if_missing=True)

image = (
    modal.Image.from_registry('nvidia/cuda:13.0.0-devel-ubuntu24.04', add_python='3.12')
    .apt_install('git', 'build-essential')
    .pip_install('torch==2.11.0', 'torchvision==0.26.0', 'ninja', 'packaging', 'setuptools', 'wheel')
    .pip_install('triton==3.7.1', extra_options='--no-deps')
    .pip_install('numpy==1.26.4', 'einops==0.8.2', 'transformers==4.57.3',
                 'pydantic==2.13.4', 'pandas==3.0.3', 'scipy==1.17.1',
                 'wandb==0.28.0', 'click', 'tqdm', 'rich', 'PyYAML',
                 'flash-linear-attention==0.5.2')
    .env({'TORCH_CUDA_ARCH_LIST': '9.0', 'MAX_JOBS': '4', 'CUDA_HOME': '/usr/local/cuda',
          'CC': 'gcc', 'CXX': 'g++', 'CUDAHOSTCXX': 'g++'})
    .env({'CAUSAL_CONV1D_FORCE_BUILD': 'TRUE', 'MAMBA_FORCE_BUILD': 'TRUE'})
    .add_local_dir(BUNDLE / 'kernel_sources', '/tmp/kernel_sources', copy=True)
    .run_commands('python -m pip install --no-build-isolation --no-deps '
                  '/tmp/kernel_sources/causal_conv1d-1.6.2.post1 '
                  '/tmp/kernel_sources/mamba_ssm-2.3.2.post1')
    .pip_install('modal==1.5.5')
    .env({'PYTHONPATH': '/opt/smat/the GPU cluster:/opt/smat/smat:/opt/zoology:/opt/fla-source',
          'FLA_TILELANG': '0', 'TRITON_F32_DEFAULT': 'tf32x3', 'WANDB_MODE': 'disabled',
          'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
          'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
          'MQAR_FOUR_READS_DIR': str(RUN)})
    .add_local_dir(BUNDLE / 'the GPU cluster', str(REMOTE))
    .add_local_dir(BUNDLE / 'smat', '/opt/smat/smat')
    .add_local_dir(BUNDLE / 'zoology', '/opt/zoology/zoology')
    .add_local_dir(BUNDLE / 'fla', '/opt/fla-source/fla')
    .add_local_dir(BUNDLE / 'zoology_cache', str(REMOTE / 'zoology_cache'))
)


def launch(d, smoke=False):
    folder = RUN / 'smoke' if smoke else RUN
    folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(ZOO_DM='16', ZOO_DS=str(d), ZOO_EPOCHS='2' if smoke else '32',
               MQAR_RUN_DIR=str(folder), MQAR_MAX_MINUTES='-1' if smoke else '330',
               TRITON_CACHE_DIR=f'/tmp/triton-d{d}')
    env.pop('ZOO_SMOKE', None)
    if smoke:
        env['ZOO_SMOKE'] = '1'
    status = folder / f'w16-d{d}.json'
    log_path = folder / f'w16-d{d}.log'
    previous = status.read_text() if status.exists() else ''
    with log_path.open('a') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'zoology.launch',
                                  str(REMOTE / 'zoo_gdn_four_reads_configs.py')],
                                 cwd=REMOTE, env=env, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            if status.exists():
                current = status.read_text()
                if current != previous:
                    previous = current
                    volume.commit()
                    print(json.dumps(dict(d=d, smoke=smoke, **json.loads(current))), flush=True)
            time.sleep(5)
        code = child.returncode
    volume.commit()
    if code:
        print(log_path.read_text()[-10000:], flush=True)
        raise RuntimeError(f'd={d} training failed with exit code {code}')
    return json.loads(status.read_text())


@app.function(image=image, gpu='H100', cpu=4, memory=32768,
              volumes={'/results': volume}, timeout=3600)
def preflight():
    volume.reload()
    RUN.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-u', '-c',
                    'from test_four_reads import check_gpu; check_gpu()'], cwd=REMOTE, check=True)
    launch(4, smoke=True)
    status = launch(4, smoke=True)
    assert status['complete']
    assert 'RESUMED' in (RUN / 'smoke/w16-d4.log').read_text()
    (RUN / 'validated.json').write_text(json.dumps(dict(gpu_checks=True, trainer_resume=True)))
    volume.commit()
    return 'GPU and trainer resume checks passed'


@app.function(image=image, gpu='H100', cpu=4, memory=32768,
              volumes={'/results': volume}, timeout=21600, max_containers=3,
              retries=modal.Retries(max_retries=1))
def train(d: int):
    assert d in (2, 3, 4)
    volume.reload()
    assert (RUN / 'validated.json').exists()
    print(f'START d={d} width=16 heads=1 head/state=16 write=1 read=4', flush=True)
    status = launch(d)
    if not status['complete']:
        raise RuntimeError(f'd={d} reached its time slice; retry resumes its checkpoint')
    return dict(d=d, **status)


@app.function(timeout=46800)
def sweep():
    print(preflight.remote(), flush=True)
    calls = {d: train.spawn(d) for d in (2, 3, 4)}
    print('TRAINING_CALLS ' + json.dumps({d: call.object_id for d, call in calls.items()}), flush=True)
    results = {d: call.get() for d, call in calls.items()}
    print('FINAL_RESULTS ' + json.dumps(results), flush=True)
    return json.dumps(results)

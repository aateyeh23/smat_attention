"""Read-only checkpoint evaluation on one separate H100; training is untouched."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import modal

LOCAL = Path(__file__).resolve().parent
data = modal.Volume.from_name('smat-pg19-scale-data-v1')
volume = modal.Volume.from_name('pg19-w768-rank64')
kernel_cache = modal.Volume.from_name('pg19-transport-optimization')
if modal.is_local():
    from modal_pg19_six import image
    image = image.add_local_file(LOCAL/'lm/eval_pg19_positions.py', '/opt/pg19-rank64/lm/eval_pg19_positions.py')
else:
    image = modal.Image.debian_slim()
app = modal.App('pg19-position-eval')


@app.function(image=image, gpu='H100', cpu=4, memory=32768, timeout=3600,
              max_containers=1, volumes={'/data':data, '/checkpoints':volume, '/eval-results':kernel_cache})
def evaluate(arms: str='gdn-baseline,mamba2-baseline,mamba2-smat-d2', tokens: int=250085376):
    data.reload()
    volume.reload()
    kernel_cache.reload()
    output = Path('/eval-results/position-eval')/str(tokens)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='/opt/pg19-rank64/lm:/opt/pg19-rank64:'+os.environ['PYTHONPATH'],
               PYTHONSAFEPATH='1', SMAT_WRITE_HASH_BACKEND='triton',
               TRITON_CACHE_DIR='/tmp/pg19-position-triton')
    archive_path = Path('/eval-results/triton-cache.tar.gz')
    if archive_path.exists():
        Path(env['TRITON_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(env['TRITON_CACHE_DIR'],filter='data')
    results = {}
    for arm in arms.split(','):
        if arm not in ('gdn-baseline','gdn-smat-d2','gdn-smat-d3','mamba2-baseline','mamba2-smat-d2','mamba2-smat-d3'):
            raise ValueError('Unknown arm')
        campaign = 'six-500m-20260917-opt-v1' if arm.startswith('gdn-smat') else 'six-500m-20260917'
        root = Path('/checkpoints/seed123')/campaign/arm
        checkpoint = root/f'model-tokens-{tokens:010d}.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        result_path = output/f'{arm}.json'
        subprocess.run([sys.executable,'-P','-u','/opt/pg19-rank64/lm/eval_pg19_positions.py',
                        '--checkpoint',str(checkpoint),'--output',str(result_path)],
                       env=env,cwd='/opt/pg19-rank64',check=True)
        kernel_cache.commit()
        results[arm] = json.loads(result_path.read_text())
    print('SUMMARY '+json.dumps({arm:row['perplexity'] for arm,row in results.items()}),flush=True)
    return results

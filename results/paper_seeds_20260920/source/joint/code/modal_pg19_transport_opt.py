"""Bounded isolated profiling; never writes active training checkpoints."""
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import modal

LOCAL=Path(__file__).resolve().parent
app=modal.App('pg19-transport-optimization')
volume=modal.Volume.from_name('pg19-transport-optimization',create_if_missing=True)
if modal.is_local():
    from modal_pg19_six import image
    image=image.add_local_file(LOCAL/'lm/bench_pg19_transport.py','/opt/pg19-rank64/lm/bench_pg19_transport.py')
    image=image.add_local_file(LOCAL/'lm/validate_pg19_transport_opt.py','/opt/pg19-rank64/lm/validate_pg19_transport_opt.py')
    image=image.add_local_file(LOCAL/'lm/bench_pg19_reader.py','/opt/pg19-rank64/lm/bench_pg19_reader.py')
    image=image.add_local_file(LOCAL/'lm/bench_pg19_write.py','/opt/pg19-rank64/lm/bench_pg19_write.py')
else:
    image=modal.Image.debian_slim()

@app.function(image=image,gpu='H100',cpu=4,memory=49152,timeout=3600,
              volumes={'/results':volume})
def run(label: str='initial', reader: bool=False, optimized: bool=False, write: bool=False):
    root=Path('/results')/label; root.mkdir(parents=True,exist_ok=True)
    cache=Path('/results/triton-cache.tar.gz')
    if cache.exists():
        Path('/tmp/pg19-triton-cache').mkdir(exist_ok=True)
        with tarfile.open(cache) as archive:archive.extractall('/tmp/pg19-triton-cache',filter='data')
    env=dict(os.environ,PYTHONPATH='/opt/pg19-rank64/lm:/opt/pg19-rank64:'+os.environ['PYTHONPATH'],
             PYTHONSAFEPATH='1',TRITON_CACHE_DIR='/tmp/pg19-triton-cache',SMAT_WRITE_HASH_BACKEND='triton')
    with (root/'bench.log').open('w') as fp:
        script=['/opt/pg19-rank64/lm/bench_pg19_reader.py'] if reader else ['/opt/pg19-rank64/lm/bench_pg19_transport.py','--output',str(root)]
        if write:script=['/opt/pg19-rank64/lm/bench_pg19_write.py']
        if optimized:script.append('--optimized')
        child=subprocess.Popen([sys.executable,'-P','-u',*script],env=env,cwd='/opt/pg19-rank64',stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True)
        for line in child.stdout:
            fp.write(line); fp.flush(); print(line,end='',flush=True)
            volume.commit()
        code=child.wait()
    with tarfile.open('/results/triton-cache.tar.gz','w:gz') as archive:
        archive.add('/tmp/pg19-triton-cache',arcname='.')
    volume.commit()
    if code: raise RuntimeError(f'Benchmark failed: {code}')
    return str(root)


@app.function(image=image,gpu='H100',cpu=4,memory=65536,timeout=3600,
              volumes={'/results':volume,'/checkpoints':modal.Volume.from_name('pg19-w768-rank64'),
                       '/data':modal.Volume.from_name('smat-pg19-scale-data-v1')})
def validate(chunk: int=16, label: str='final'):
    root=Path('/results')/f'validate-{label}-chunk{chunk}';root.mkdir(parents=True,exist_ok=True)
    cache=Path('/results/triton-cache.tar.gz')
    if cache.exists():
        Path('/tmp/pg19-triton-cache').mkdir(exist_ok=True)
        with tarfile.open(cache) as archive:archive.extractall('/tmp/pg19-triton-cache',filter='data')
    env=dict(os.environ,PYTHONPATH='/opt/pg19-rank64/lm:/opt/pg19-rank64:'+os.environ['PYTHONPATH'],
             PG19_MEMORY_VARIANT='updated_transport',PYTHONSAFEPATH='1',
             TRITON_CACHE_DIR='/tmp/pg19-triton-cache',SMAT_WRITE_HASH_BACKEND='triton')
    with (root/'validation.log').open('w') as fp:
        child=subprocess.Popen([sys.executable,'-P','-u','/opt/pg19-rank64/lm/validate_pg19_transport_opt.py',
            '--output',str(root),'--chunk',str(chunk)],env=env,cwd='/opt/pg19-rank64',
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in child.stdout:
            fp.write(line);fp.flush();print(line,end='',flush=True);volume.commit()
        code=child.wait()
    volume.commit()
    if code:raise RuntimeError(f'Full validation failed: {code}')
    return str(root)

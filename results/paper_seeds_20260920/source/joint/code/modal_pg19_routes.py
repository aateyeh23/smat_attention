"""Small read-only SMAT routing probe, independent of active training jobs."""
import json,os,subprocess,sys,tarfile,time
from pathlib import Path
import modal
LOCAL=Path(__file__).resolve().parent
app=modal.App('pg19-routing-probe')
checkpoints=modal.Volume.from_name('pg19-w768-rank64')
data=modal.Volume.from_name('smat-pg19-scale-data-v1')
cache=modal.Volume.from_name('pg19-transport-optimization')
results=modal.Volume.from_name('pg19-routing-probes',create_if_missing=True)
if modal.is_local():
    from modal_pg19_1b import image
    image=image.add_local_file(LOCAL/'lm/probe_pg19_routes.py','/opt/pg19-rank64/lm/probe_pg19_routes.py')
else:image=modal.Image.debian_slim()

@app.function(image=image,gpu='H100',cpu=4,memory=49152,timeout=3600,max_containers=4,
              volumes={'/checkpoints':checkpoints,'/data':data,'/cache':cache,'/results':results})
def probe(arm:str):
    checkpoints.reload();data.reload();cache.reload()
    source=None
    for c in ['six-1b-constant3e-5-20260917','six-750m-constant3e-5-20260917']:
        p=Path('/checkpoints/seed123')/c/arm/'latest.pt'
        if p.exists():source=p;break
    if source is None:raise FileNotFoundError(arm)
    env=dict(os.environ,PYTHONPATH='/opt/pg19-rank64/lm:/opt/pg19-rank64:'+os.environ['PYTHONPATH'],
        PYTHONSAFEPATH='1',SMAT_WRITE_HASH_BACKEND='triton',TRITON_CACHE_DIR='/tmp/route-probe-triton')
    if Path('/cache/triton-cache.tar.gz').exists():
        Path(env['TRITON_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
        with tarfile.open('/cache/triton-cache.tar.gz') as f:f.extractall(env['TRITON_CACHE_DIR'],filter='data')
    out=Path('/results')/arm;out.mkdir(parents=True,exist_ok=True)
    with (out/'probe.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-P','-u','/opt/pg19-rank64/lm/probe_pg19_routes.py',
            '--checkpoint',str(source),'--output',str(out/'probe.json')],env=env,cwd='/opt/pg19-rank64',
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in child.stdout:
            log.write(line);log.flush()
            if not line.startswith('RESULT '):print(arm+' '+line,end='',flush=True)
        code=child.wait()
    results.commit()
    if code:raise RuntimeError(f'{arm} probe exited {code}')
    r=json.loads((out/'probe.json').read_text())
    summary=dict(arm=arm,tokens=r['tokens'],ablation=r['ablation'],
        relative_rms=[x['relative_rms'] for x in r['layers']],gate_means=[x['gate']['mean'] for x in r['layers']])
    print('SUMMARY '+json.dumps(summary),flush=True)
    return summary

@app.function(timeout=4000)
def sweep():
    calls={a:probe.spawn(a) for a in ['gdn-smat-d2','gdn-smat-d3','mamba2-smat-d2','mamba2-smat-d3']}
    print('CALLS '+json.dumps({a:c.object_id for a,c in calls.items()}),flush=True)
    out={}
    for a,c in calls.items():
        try:out[a]=c.get()
        except Exception as e:out[a]=dict(error=str(e))
    print('RESULTS '+json.dumps(out),flush=True)
    return out

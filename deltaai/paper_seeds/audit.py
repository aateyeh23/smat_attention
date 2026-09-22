"""Audit the requested matrix, completed training budgets, and aggregation."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent/'results/paper_seeds_20260920'


def read(path):
    return json.loads(path.read_text())


def finite(value):
    if isinstance(value,float):assert math.isfinite(value)
    elif isinstance(value,dict):
        for x in value.values():finite(x)
    elif isinstance(value,list):
        for x in value:finite(x)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    tasks=read(ROOT/'tasks.json')
    expected=set()
    for seed in [1,2,3,4]:
        for family in ['gdn','mamba2']:
            for width in [16,32,64]:
                for d in [0,1,2,3,4]:expected.add(('mqar',family,width,d,seed))
        for family in ['mamba2','gdn_current']:
            for d in [1,2,3,4]:expected.add(('joint',family,64,d,seed))
    for seed in [123,1,2,3,4]:
        for family in ['mamba2','gdn']:expected.add(('joint_loglinear',family,64,0,seed))
    actual=[tuple(t[k] for k in ['task','family','width','d','seed']) for t in tasks]
    assert len(actual)==len(set(actual))==162 and set(actual)==expected
    complete=[];missing=[]
    data_sha=hashlib.sha256((ROOT/'joint/data/manifest.json').read_bytes()).hexdigest()
    for task in tasks:
        path=Path(task['result']);folder=path.parent
        if not path.exists():missing.append(task['name']);continue
        result=read(path);assert result['complete'],path;finite(result)
        if task['task']=='mqar':
            config=read(folder/'config.json')
            assert config['seed']==task['seed'] and config['max_epochs']==32
            assert config['learning_rate']==task['lr'] and config['weight_decay']==.1
            assert config['early_stopping_metric'] is None
            assert config['model']['d_model']==task['width'] and config['model']['n_layers']==2
            assert config['data']['seed']==123 and config['data']['batch_size']==[256,32]
            mixer=config['model']['sequence_mixer'];d=task['d']
            if d==0:
                assert mixer['name']=='zoo_log_linear.LogLinearMixer'
                assert mixer['kwargs']==dict(family=task['family'])
            elif task['family']=='mamba2':
                assert mixer['name']=='zoo_mamba_four_reads.MqarMambaFourReads' and mixer['kwargs']['d']==d
            else:
                assert mixer['kwargs']['d']==d
                assert mixer['kwargs']['n_heads']==(1 if task['width']==16 else 2)
                transport=task['width']>=32 and d>=2
                assert mixer['name']==('zoo_gdn_transport.SmatGDNTransport' if transport else 'zoo_smat_gdn.SmatGDNReset')
                if transport:
                    for key,value in dict(write_hash_neighbor_grad=True,transport_scalar_decay=True,
                                          detach_write_hash_input=True,memory_plant=False,memory_incidence_rescale=False).items():
                        assert mixer['kwargs'][key]==value
            stem=f"w{task['width']}-d{max(1,d)}"
            history=[json.loads(line) for line in (folder/(stem+'-history.jsonl')).read_text().splitlines()]
            assert {r['epoch'] for r in history}==set(range(1,33)),folder
            assert result['next_epoch']==32 and history[-1]['epoch']==32
            for key,value in result['metrics'].items():assert history[-1]['metrics'][key]==value
            assert (folder/(stem+'.pt')).is_file()
        else:
            recipe=read(folder/'recipe.json')
            for key in ['family','width','d','seed','lr']:assert recipe[key]==task[key]
            assert recipe['batch']==256 and recipe['epochs']==32 and recipe['total_steps']==22560
            assert recipe['dataset_manifest_sha256']==data_sha and recipe['early_stopping'] is False
            assert recipe['lengths']==[64,100,772,1540,3076]
            assert result['epochs']==32 and result['steps']==22560 and result['examples_seen']==5760000
            history=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in history]==list(range(1,33)),folder
            assert history[-1]['steps']==22560 and (folder/'checkpoint.pt').is_file()
            for metrics in [history[-1],result['final_test']]:
                assert set(metrics['cells'])=={'c1-k4','c2-k8','c8-k16','c16-k16','c32-k16'}
                assert abs(metrics['accuracy']-sum(c['accuracy'] for c in metrics['cells'].values())/5)<1e-12
            if task['task']=='joint_loglinear':
                extra=read(folder/'loglinear_recipe.json')
                assert extra['levels']==13 and extra['logical_head_and_state_dim']==16
                assert extra['source_sha256']==read(ROOT/'loglinear_validated.json')['sha256']
        finite(history);complete.append(task['name'])
    counts=collections.Counter(t['task'] for t in tasks if t['name'] in complete)
    summary=read(ROOT/'summary.json')
    if not missing:
        rows=[r for r in summary if r['split']=='validation' and r['cell']=='macro']
        assert len(rows)==40
        for row in rows:assert row['n']==5 and set(row['seeds'])=={123,1,2,3,4}
    report=dict(requested_runs=162,completed_runs=len(complete),counts=dict(counts),
                completed_checks_passed=True,complete=not missing,missing=missing)
    (ROOT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='missing'}))
    if args.require_complete and missing:raise SystemExit(10)


if __name__=='__main__':main()

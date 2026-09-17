"""One Log-Linear MQAR arm on SCF: the body of modal_log_linear.train, on Slurm.

Manifest fields, the source-SHA gate against the validation pass, the per-epoch
checkpoint snapshots and result.json are all as in modal_log_linear.py; the
volume commits are dropped because the results directory is already on disk.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL / 'results/mqar_log_linear'


def manifest_for(family, width, sha):
    return dict(family=family, width=width, seed=123,
        lr=.003 if family == 'gdn' and width == 64 else .01,
        layers=2, epochs=32, head_dim=16, state_dim=16,
        heads=(1 if width == 16 else 2) if family == 'gdn' else width // 8,
        backend='dense-pytorch-exact-log-linear', hierarchy='base2-WEAK', levels=9,
        upstream_commit='7f8644159c1406fae1ad863829a5b3a4fbf63022', sha256=sha,
        lambda_mode='softplus(L * l_proj(u))', weight_decay=.1, precision='float32',
        note='Matched Zoology backbone; MQAR levels9 rather than unused upstream LM levels15; no epoch gates')


def execute(cmd, env, root, name, width):
    with (root / name).open('a') as log:
        child = subprocess.Popen(cmd, cwd=str(LOCAL), env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
        for line in child.stdout:
            log.write(line); log.flush()
            if any(m in line for m in ('PASS', 'CHECKPOINT', 'SOURCE_AUDIT', 'TRAINING',
                                       'OPTIMIZER_CONFIG', 'INTERLEAVED')):
                print(line, end='', flush=True)
            if 'CHECKPOINT epoch=' in line:
                epoch = int(line.split('epoch=', 1)[1].split()[0])
                snapshots = root / 'checkpoints'; snapshots.mkdir(exist_ok=True)
                shutil.copy2(root / f'w{width}-d1.pt', snapshots / f'epoch{epoch:02d}.pt')
        code = child.wait()
    if code:
        print((root / name).read_text()[-10000:], flush=True)
        raise SystemExit(f'{name} exit {code}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('family', choices=('gdn', 'mamba2'))
    p.add_argument('width', type=int, choices=(16, 32, 64))
    p.add_argument('--max-minutes', default=os.environ.get('MQAR_MAX_MINUTES', '690'))
    args = p.parse_args()

    root = ROOT / f'{args.family}-w{args.width}'; root.mkdir(parents=True, exist_ok=True)
    source = Path(os.environ['LL_SOURCE_PATH'])
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    gate = json.loads((ROOT / 'validation/passed.json').read_text())
    assert gate['sha256'] == sha and gate['passed'], 'GPU validation gate does not cover this source'

    manifest = manifest_for(args.family, args.width, sha)
    path = root / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'manifest changed under an existing run'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    sources = root / 'sources'; sources.mkdir(exist_ok=True)
    for name in ('zoo_log_linear.py', 'zoo_log_linear_configs.py', 'zoo_log_linear_block.py',
                 'zoo_mqar_resume.py', 'zoo_mqar_interleave.py', 'zoo_reference_configs.py',
                 'test_log_linear_gpu.py'):
        shutil.copy2(LOCAL / name, sources / name)

    env = dict(os.environ, LL_FAMILY=args.family, LL_SOURCE_SHA256=sha,
               ZOO_DM=str(args.width), ZOO_DS='1', ZOO_EPOCHS='32',
               MQAR_RUN_DIR=str(root), MQAR_MAX_MINUTES=str(args.max_minutes))
    env.pop('ZOO_SMOKE', None)
    execute([sys.executable, '-u', '-m', 'zoology.launch', str(LOCAL / 'zoo_log_linear_configs.py')],
            env, root, 'train.log', args.width)

    status = json.loads((root / f'w{args.width}-d1.json').read_text())
    (root / 'result.json').write_text(json.dumps(status, indent=2) + '\n')
    print('ARM ' + json.dumps(dict(family=args.family, width=args.width,
                                   complete=status['complete'], next_epoch=status['next_epoch'],
                                   accuracy=status['metrics']['valid/accuracy'])), flush=True)
    raise SystemExit(0 if status['complete'] else 3)   # 3: time limit, resubmit to resume


if __name__ == '__main__':
    main()

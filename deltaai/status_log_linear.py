"""Status of the six MQAR Log-Linear arms, read from disk.

collect_log_linear.py pulls the same fields out of a Modal volume; on SCF the
arms write straight into results/mqar_log_linear/, so there is nothing to pull.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'results/mqar_log_linear'
CELLS = ['valid/num_kv_pairs/accuracy-4', 'valid/num_kv_pairs/accuracy-8',
         'valid/num_kv_pairs/accuracy-16', 'valid/num_kv_pairs/accuracy-32',
         'valid/num_kv_pairs/accuracy-64']


def row(family, width):
    name = f'{family}-w{width}'
    status = ROOT / name / f'w{width}-d1.json'
    manifest = ROOT / name / 'manifest.json'
    lr = json.loads(manifest.read_text())['lr'] if manifest.exists() else None
    if not status.exists():
        return dict(run=name, lr=lr, status='no checkpoint yet')
    s = json.loads(status.read_text())
    m = s['metrics']
    return dict(run=name, lr=lr, epoch=s['next_epoch'], complete=bool(s['complete']),
                accuracy=round(m['valid/accuracy'], 4),
                cells=[round(m[k], 4) for k in CELLS if k in m])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--json', action='store_true')
    args = p.parse_args()
    rows = [row(f, w) for f in ('gdn', 'mamba2') for w in (16, 32, 64)]
    (ROOT / 'status.json').write_text(json.dumps(rows, indent=2) + '\n')
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"{'run':<14}{'lr':>7}{'epoch':>7}{'done':>6}{'mean':>8}   per-cell 4/8/16/32/64")
        for r in rows:
            if 'status' in r:
                print(f"{r['run']:<14}{str(r['lr']):>7}   {r['status']}")
            else:
                cells = ' '.join(f'{c:.3f}' for c in r['cells'])
                print(f"{r['run']:<14}{r['lr']:>7}{r['epoch']:>7}{str(r['complete']):>6}"
                      f"{r['accuracy']:>8.4f}   {cells}")

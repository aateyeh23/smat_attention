"""The six Log-Linear MQAR arms as one CSV, in percent, per cell and mean.

Columns follow results/gdn_smat_summary/mqar.csv: one row per arm, the five
in-distribution test cells (input_seq_len, num_kv_pairs) = (64,4) (64,8)
(64,16) (128,32) (256,64), and the mean over them that the Log-Linear table
reports. The 512/128 and 1024/256 length-generalisation cells are excluded, as
in that paper; the training mixture is the separate five-cell set in
zoo_reference_configs.py. Columns are named by num_kv_pairs, which is what
Zoology slices on, and each cell contributes 1000 examples, so the mean is
unweighted across cells.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'results/mqar_log_linear'
PAIRS = (4, 8, 16, 32, 64)
FIELDS = ['family', 'width', 'heads', 'head_dim', 'state_dim', 'lr', 'seed', 'epoch',
          'complete', 'mean'] + [f'kv{p}' for p in PAIRS]


def rows():
    for family in ('gdn', 'mamba2'):
        for width in (16, 32, 64):
            run = ROOT / f'{family}-w{width}'
            manifest = json.loads((run / 'manifest.json').read_text())
            status = json.loads((run / f'w{width}-d1.json').read_text())
            m = status['metrics']
            yield dict(family=family, width=width, heads=manifest['heads'],
                       head_dim=manifest['head_dim'], state_dim=manifest['state_dim'],
                       lr=manifest['lr'], seed=manifest['seed'],
                       epoch=status['next_epoch'], complete=bool(status['complete']),
                       mean=round(100 * m['valid/accuracy'], 2),
                       **{f'kv{p}': round(100 * m[f'valid/num_kv_pairs/accuracy-{p}'], 2)
                          for p in PAIRS})


if __name__ == '__main__':
    table = list(rows())
    out = ROOT / 'mqar_log_linear.csv'
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, FIELDS); writer.writeheader(); writer.writerows(table)
    head = f"{'arm':<12}{'lr':>7}{'heads':>6}{'ep':>4}{'done':>6}{'mean':>8}" + \
           ''.join(f'{"kv"+str(p):>8}' for p in PAIRS)
    print(head); print('-' * len(head))
    for r in table:
        print(f"{r['family']+'-w'+str(r['width']):<12}{r['lr']:>7}{r['heads']:>6}"
              f"{r['epoch']:>4}{str(r['complete']):>6}{r['mean']:>8.2f}"
              + ''.join(f"{r['kv'+str(p)]:>8.2f}" for p in PAIRS))
    print(f'\nwrote {out}')

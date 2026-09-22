"""Download committed MQAR metrics and render an up-to-date comparison table."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
from datetime import datetime, timezone
import json
import time
from pathlib import Path
import modal

OUT = Path(__file__).resolve().parent/'results/mqar_width_sweeps'


def collect():
    volume = modal.Volume.from_name('smat-mqar-width-sweeps')
    arms = [(family, width, d) for family, widths in [('gdn', (32,64)), ('mamba2',(16,32,64))]
            for width in widths for d in (1,2,3,4)]

    def fetch(arm):
        family, width, d = arm
        name = f'w{width}-d{d}'
        root = OUT/family
        root.mkdir(parents=True, exist_ok=True)
        result = dict(family=family, width=width, d=d, epoch=0, complete=False,
                      accuracy=None, accuracy_64_pairs=None, loss=None)
        for suffix in ('.json', '-history.jsonl'):
            try:
                raw = b''.join(volume.read_file(f'/seed123/full/{family}/{name}{suffix}'))
            except FileNotFoundError:
                continue
            (root/(name+suffix)).write_bytes(raw)
            if suffix == '.json':
                obj = json.loads(raw)
                metrics = obj['metrics']
                result.update(epoch=obj['next_epoch'], complete=obj['complete'],
                    accuracy=100*metrics['valid/accuracy'],
                    accuracy_64_pairs=100*metrics['valid/num_kv_pairs/accuracy-64'],
                    loss=metrics['valid/loss'])
        checkpoint = root/(name+'.pt')
        if result['complete'] and not checkpoint.exists():
            raw = b''.join(volume.read_file(f'/seed123/full/{family}/{name}.pt'))
            temporary = checkpoint.with_suffix('.tmp')
            temporary.write_bytes(raw)
            temporary.replace(checkpoint)
        return result

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(fetch, arms))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(dict(updated=datetime.now(timezone.utc).isoformat(), runs=rows), indent=2)+'\n')
    with (OUT/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ['MQAR width sweeps — seed 123, 32 epochs; learned 1-write/4-read SMAT.',
             '', 'GDN has 2 heads. Mamba2 uses its native 2× expansion: 2/4/8 heads at width 16/32/64. Head and state dimensions are 16.',
             '', '| Family | Width | Variant | Epoch | Accuracy | 64-pair accuracy | Loss |',
             '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        variant = 'Baseline' if r['d']==1 else 'SMAT d='+str(r['d'])
        metric = lambda key: '—' if r[key] is None else f'{r[key]:.2f}'
        table.append(f"| {r['family']} | {r['width']} | {variant} | {r['epoch']}/32 | {metric('accuracy')}% | {metric('accuracy_64_pairs')}% | {metric('loss')} |")
    (OUT/'report.md').write_text('\n'.join(table)+'\n')
    print(json.dumps(rows), flush=True)
    return all(row['complete'] for row in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        done = collect()
        if done or not args.watch:
            break
        time.sleep(60)

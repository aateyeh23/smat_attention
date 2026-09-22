"""Fetch lightweight training/validation logs; large checkpoints stay on Modal."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import modal

OUT = Path(__file__).resolve().parent/'results/pg19_updated_transport'


def collect():
    volume = modal.Volume.from_name('pg19-updated-transport-d34')
    def fetch(d):
        folder = OUT/f'd{d}'
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(d=d, complete=False, tokens=0, _training={})
        for name in ('recipe.json', 'status.json', 'metrics.jsonl'):
            try:
                raw = b''.join(volume.read_file(f'/seed123/d{d}/{name}'))
            except FileNotFoundError:
                continue
            (folder/name).write_bytes(raw)
            if name == 'status.json':
                row.update(json.loads(raw))
            if name == 'metrics.jsonl':
                for line in raw.decode().splitlines():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row['tokens'] = max(row['tokens'], item['tokens'])
                    if item['event'] == 'train':
                        row['train'] = item
                        row['_training'][item['tokens']] = item
                    elif item['event'] == 'validation':
                        row['validation'] = item
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, (3,4)))
    histories = [r.pop('_training') for r in rows]
    common_tokens = set.intersection(*(set(h) for h in histories))
    shared = max(common_tokens) if common_tokens else None
    matched = {r['d']: h[shared] for r,h in zip(rows,histories)} if shared else None
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(dict(updated=datetime.now(timezone.utc).isoformat(), runs=rows,
                                                   matched_train=matched), indent=2)+'\n')
    lines = ['PG19 — 2B unique training tokens per model, 16 layers, width 1024, 16K context.',
             '', 'Shared effective batch: 131,072 tokens. Microbatches: 3 + 3 + 2 sequences. Seed 123.',
             'Checkpoints: Modal volume `pg19-updated-transport-d34`, `/seed123/d{3,4}/`.',
             '', '| Model | Tokens (M) | Train loss | Validation perplexity | Recent tokens/s | Complete |',
             '|---|---:|---:|---:|---:|---|']
    for row in rows:
        train, valid = row.get('train', {}), row.get('validation') or {}
        loss = f"{train['loss']:.3f}" if train else '—'
        ppl = f"{valid['perplexity']:.2f}" if 'perplexity' in valid else '—'
        speed = f"{train['tokens_per_second']:,.0f}" if train else '—'
        label = 'GDN' if row['d'] == 1 else f"GDN + SMAT d={row['d']}"
        lines.append(f"| {label} | {row['tokens']/1e6:.2f} | {loss} | {ppl} | {speed} | {row['complete']} |")
    if matched:
        lines += ['', f'Latest shared training point: **{shared/1e6:.2f}M tokens**, identical data batches.',
                  '', '| Model | Matched training loss |', '|---|---:|']
        for d,item in matched.items():
            label = 'GDN' if d == 1 else f'GDN + SMAT d={d}'
            lines.append(f"| {label} | {item['loss']:.3f} |")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows), flush=True)
    return all(r['complete'] for r in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        complete = collect()
        if complete or not args.watch:
            break
        time.sleep(60)

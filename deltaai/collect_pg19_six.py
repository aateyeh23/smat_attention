"""Collect six PG19 arms and compare validation at a common training-token count."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

import modal

OUT = Path(__file__).resolve().parent/'results/pg19_six_500m'
ARMS = [f'{f}-{suffix}' for f in ('gdn','mamba2')
        for suffix in ('baseline','smat-d2','smat-d3')]


def collect(campaign, output):
    volume = modal.Volume.from_name('pg19-w768-rank64')

    def fetch(name):
        folder = output/name
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(arm=name, tokens=0, phase='pending', complete=False)
        valid = {}
        arm_campaign=campaign
        try:
            raw=b''.join(volume.read_file(f'/seed123/{campaign}/{name}/continuation.json'))
        except (FileNotFoundError,modal.exception.NotFoundError):
            pass
        else:
            continuation=json.loads(raw)
            arm_campaign=continuation['campaign']
            if Path(arm_campaign).name!=arm_campaign or continuation['arm']!=name:
                raise ValueError('Invalid continuation record')
            (folder/'continuation.json').write_bytes(raw)
            row['continuation_campaign']=arm_campaign
        for filename in ('manifest.json','recipe.json','run_state.json','validated.json',
                         'status.json','metrics.jsonl','migration.json'):
            try:
                raw = b''.join(volume.read_file(f'/seed123/{arm_campaign}/{name}/{filename}'))
            except (FileNotFoundError, modal.exception.NotFoundError):
                continue
            (folder/filename).write_bytes(raw)
            if filename == 'run_state.json':
                state = json.loads(raw)
                row['phase'] = state['state']
                if 'error' in state:
                    row['error'] = state['error']
            elif filename == 'status.json':
                row.update(json.loads(raw))
                row['saved_tokens'] = row['tokens']
            elif filename == 'manifest.json':
                manifest = json.loads(raw)
                row.update(target_tokens=manifest['target_tokens'], microbatch=manifest['batch'])
                row['kernel_optimization']=manifest.get('kernel_optimization')
            elif filename == 'metrics.jsonl':
                for line in raw.decode().splitlines():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row['tokens'] = max(row['tokens'], item['tokens'])
                    if item['event'] == 'train':
                        row['train'] = item
                    elif item['event'] == 'validation':
                        row['validation'] = item
                        valid[(item['tokens'], item['full'])] = item
        return row, valid

    with ThreadPoolExecutor(max_workers=6) as pool:
        fetched = list(pool.map(fetch, ARMS))
    rows, histories = zip(*fetched)
    common = set.intersection(*(set(h) for h in histories))
    latest = max(common) if common else None
    matched = {row['arm']:hist[latest] for row,hist in zip(rows,histories)} if latest else None
    summary = dict(updated=datetime.now(timezone.utc).isoformat(), campaign=campaign,
                   runs=rows, matched_validation=matched)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# PG19 six-run comparison', '', f"Updated: {summary['updated']}", '',
        '16 layers, width 768, FFN 2048, 16K context. One H100 per arm. Seed 123.',
        'Plain baselines retain full-sequence recurrence; SMAT arms reset at the midpoint.', '',
        '| Model | Phase | Tokens (M) | Train loss | Latest val PPL (tokens M) | Tokens/s |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        train = row.get('train', {})
        valid = row.get('validation') or {}
        loss = f"{train['loss']:.3f}" if train else '—'
        speed = f"{train['tokens_per_second']:,.0f}" if train else '—'
        ppl = (f"{valid['perplexity']:.2f} ({valid['tokens']/1e6:.2f})"
               if 'tokens' in valid else '—')
        lines.append(f"| {row['arm']} | {row['phase']} | {row['tokens']/1e6:.2f} | {loss} | {ppl} | {speed} |")
        if 'error' in row:
            lines.append(f"\n{row['arm']} error: {row['error']}\n")
    if matched:
        lines += ['', f'## Validation at {latest[0]/1e6:.2f}M matched training tokens', '',
                  '| Model | Perplexity |', '|---|---:|']
        lines += [f"| {name} | {item['perplexity']:.2f} |" for name,item in matched.items()]
    else:
        lines += ['', 'No validation point shared by all six arms yet.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(updated=summary['updated'],
        runs=[{k:r[k] for k in ('arm','phase','tokens')} for r in rows])), flush=True)
    return all(r['phase'] in ('complete','paused','failed') for r in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', default='six-500m-20260917')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'collector.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                done = collect(args.campaign, args.output)
            except Exception as error:
                print(f'COLLECTION_ERROR {type(error).__name__}: {error}', flush=True)
                if not args.watch:
                    raise
                done = False
            if done or not args.watch:
                break
            time.sleep(60)

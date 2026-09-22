"""Download RULER progress, predictions and provenance without touching training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import modal

OUT = Path(__file__).resolve().parent/'results/pg19_ruler_mk_niah1'
ARMS = [f'{family}-{suffix}' for family in ('gdn','mamba2')
        for suffix in ('baseline','smat-d2','smat-d3')]


def collect():
    volume = modal.Volume.from_name('pg19-ruler-evaluation')
    def fetch(arm):
        folder = OUT/arm
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(arm=arm, state='pending', completed=0, total=500)
        variant = 'full'
        try:
            b''.join(volume.read_file(f'/mk-niah1-16k-500m/fast/{arm}/status.json'))
            variant = 'fast'
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass
        if (OUT/'recurrent-launched.json').exists():
            variant = 'recurrent'
        row['decoder_variant'] = variant
        for name in ('run_state.json', 'status.json', 'manifest.json', 'predictions.jsonl', 'result.json', 'eval.log'):
            try:
                raw = b''.join(volume.read_file(f'/mk-niah1-16k-500m/{variant}/{arm}/{name}'))
            except (FileNotFoundError, modal.exception.NotFoundError):
                continue
            (folder/name).write_bytes(raw)
            if name in ('run_state.json', 'status.json', 'result.json'):
                row.update(json.loads(raw))
        if ((OUT/'paused-for-recurrent-decoding.json').exists()
                and not (OUT/'recurrent-launched.json').exists()):
            row['state'] = 'paused for decoder implementation'
        return row
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(fetch, ARMS))
    summary = dict(updated=datetime.now(timezone.utc).isoformat(), runs=rows)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# RULER MK-NIAH-1: PG19 500M checkpoints', '', f"Updated: {summary['updated']}", '',
             'Official task generator and substring-recall scoring. GPT-2 tokenizer, base prompt,',
             '16,384-token budget including up to 128 greedy output tokens. Same 500 examples for all arms.',
             'Models retain their trained fixed 16K geometry. No fine-tuning or checkpoint updates.', '',
             '| Model | State | Examples | Accuracy (%) |', '|---|---|---:|---:|']
    for row in rows:
        score = f"{row['score']:.2f}" if 'score' in row else '—'
        lines.append(f"| {row['arm']} | {row['state']} | {row['completed']}/{row['total']} | {score} |")
        if 'error' in row:
            lines.extend(['', row['arm']+': '+row['error'], ''])
    finished = all(row['state'] == 'complete' for row in rows)
    lines += ['', 'Decoding uses a prompt prefill followed by cached recurrent updates. Earlier',
              'full-window and speculative outputs are excluded from these scores.', '',
              ('All six runs are complete. These are PG19 base models,' if finished else
               'Scores are provisional until all examples finish. These are PG19 base models,'),
              'so failures can reflect prompt-following ability as well as retrieval.', '',
              '[Official RULER](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)',
              '[Cached evaluation app](https://modal.com/apps/archerdwang/main/ap-q0XZKzTUemQmrdE6QRN2pu)',
              '[Mamba-2 d3 continuation](https://modal.com/apps/archerdwang/main/ap-R3QklB1SdwIR6hoaeMjldT)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary), flush=True)
    return all(row['state'] in ('complete', 'failed') for row in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            done = collect()
        except Exception as error:
            print(f'COLLECTION_ERROR {type(error).__name__}: {error}', flush=True)
            if not args.watch:
                raise
            done = False
        if done or not args.watch:
            break
        time.sleep(60)

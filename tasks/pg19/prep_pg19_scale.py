"""Prepare disjoint official PG19 splits as complete 16K next-token windows.

Each row contains length+1 tokens from one book. Train rows are subsequently
shuffled identically for all model arms. Existing local PG19 data is untouched.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import requests
import tiktoken


def prepare(folder, target_tokens=1_000_000_000, length=16384, workers=16, extend_from=None):
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.json'
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if result['target_tokens'] != target_tokens or result['length'] != length:
            raise ValueError('Existing dataset recipe differs')
        return result
    enc = tiktoken.get_encoding('gpt2')
    lists = {}
    for split in ('train', 'validation', 'test'):
        url = f'https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/{split}_files.txt'
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        lists[split] = sorted(response.text.splitlines())
        (out / f'{split}_files.txt').write_text(response.text)
    ids = {s: {Path(p).stem for p in ps} for s, ps in lists.items()}
    assert not (ids['train'] & ids['validation'] or ids['train'] & ids['test'] or ids['validation'] & ids['test'])
    random.Random(123).shuffle(lists['train'])

    def fetch(path):
        for attempt in range(5):
            try:
                r = requests.get('https://storage.googleapis.com/deepmind-gutenberg/' + path, timeout=90)
                r.raise_for_status()
                text = r.content.decode('utf-8')
                return path, enc.encode_ordinary(text) + [enc.eot_token]
            except (requests.RequestException, UnicodeError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    result = dict(target_tokens=target_tokens, length=length, tokenizer='gpt2',
                  vocab=enc.n_vocab, dtype='uint16', row_width=length+1,
                  seed=123, windows_cross_books=False, splits={})
    base = Path(extend_from) if extend_from else None
    previous = json.loads((base/'manifest.json').read_text()) if base else None
    if previous and (previous['length'] != length or previous['seed'] != 123
                     or previous['tokenizer'] != 'gpt2'
                     or previous['target_tokens'] >= target_tokens):
        raise ValueError('Incompatible source dataset for extension')
    for split in ('validation', 'test', 'train'):
        name = 'val' if split == 'validation' else split
        if base and split != 'train':
            for suffix in ('.bin', '-books.json'):
                shutil.copyfile(base/(name+suffix), out/(name+suffix))
            result['splits'][name] = previous['splits'][name]
            continue
        target_rows = math.ceil(target_tokens/length) if split == 'train' else None
        rows = 0
        books = []
        digest = hashlib.sha256()
        started = time.monotonic()
        paths = lists[split]
        skipped_rows = 0
        if base:
            books = json.loads((base/'train-books.json').read_text())
            rows = previous['splits']['train']['rows']
            skipped_rows = books[-1]['rows']
            start_book = next(i for i, p in enumerate(paths) if Path(p).stem == books[-1]['id'])
            paths = paths[start_book:]
        # Limit outstanding books; do not buffer the whole corpus in RAM.
        with (out / f'{name}.bin.partial').open('wb') as fp, ThreadPoolExecutor(max_workers=workers) as pool:
            if base:
                with (base/'train.bin').open('rb') as source:
                    while raw := source.read(8*1024*1024):
                        fp.write(raw)
                        digest.update(raw)
                if digest.hexdigest() != previous['splits']['train']['sha256']:
                    raise ValueError('Source dataset checksum mismatch')
            for offset in range(0, len(paths), workers):
                for path, tokens in pool.map(fetch, paths[offset:offset+workers]):
                    count = (len(tokens)-1)//length - skipped_rows
                    if target_rows is not None:
                        count = min(count, target_rows-rows)
                    if count:
                        arr = np.asarray(tokens, dtype=np.uint16)
                        for i in range(skipped_rows, skipped_rows+count):
                            raw = arr[i*length:i*length+length+1].tobytes()
                            fp.write(raw)
                            digest.update(raw)
                        if skipped_rows:
                            books[-1]['rows'] += count
                        else:
                            books.append(dict(id=Path(path).stem, first_row=rows, rows=count,
                                              book_tokens=len(tokens)))
                        rows += count
                    skipped_rows = 0
                    if target_rows is not None and rows >= target_rows:
                        break
                print(json.dumps(dict(event='data_progress', split=name, rows=rows,
                                      prediction_tokens=rows*length, books=len(books),
                                      seconds=time.monotonic()-started)), flush=True)
                if target_rows is not None and rows >= target_rows:
                    break
        if target_rows is not None and rows < target_rows:
            raise RuntimeError('Insufficient unique tokens in training split')
        (out / f'{name}.bin.partial').replace(out / f'{name}.bin')
        (out / f'{name}-books.json').write_text(json.dumps(books, indent=2)+'\n')
        result['splits'][name] = dict(rows=rows, prediction_tokens=rows*length,
                                     books=len(books), sha256=digest.hexdigest())
    manifest.write_text(json.dumps(result, indent=2)+'\n')
    print('DATA_READY ' + json.dumps(result), flush=True)
    return result

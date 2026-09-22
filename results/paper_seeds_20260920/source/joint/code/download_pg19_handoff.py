"""Download immutable PG19 handoff snapshots and verify every SHA-256."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import modal

OUT = Path(__file__).resolve().parent / 'results/pg19_2b_pretrain/handoff'


def download(d):
    volume = modal.Volume.from_name('smat-pg19-2b-checkpoints')
    remote = f'/seed123/d{d}/handoff'
    folder = OUT / f'd{d}'
    folder.mkdir(parents=True, exist_ok=True)
    raw = b''.join(volume.read_file(f'{remote}/manifest.json'))
    manifest = json.loads(raw)
    for name, expected in manifest['files'].items():
        destination = folder / name
        if destination.exists():
            with destination.open('rb') as stream:
                existing = hashlib.file_digest(stream, 'sha256').hexdigest()
            if existing == expected['sha256']:
                continue
        temporary = destination.with_suffix(destination.suffix + '.partial')
        digest = hashlib.sha256()
        size = 0
        with temporary.open('wb') as stream:
            for chunk in volume.read_file(f'{remote}/{name}'):
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            raise RuntimeError(f'Integrity check failed: d{d}/{name}')
        temporary.replace(destination)
        print(f'VERIFIED d{d}/{name}: {size:,} bytes', flush=True)
    (folder / 'manifest.json').write_bytes(raw)
    return manifest


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, (1, 2, 3, 4)))
    (OUT / 'checkpoints.json').write_text(json.dumps(records, indent=2) + '\n')
    print('ALL FOUR HANDOFFS VERIFIED', flush=True)

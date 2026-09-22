"""Isolated joint-recall incidence experiment, with frozen training sources."""
import hashlib
import json
import os
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent / 'results/joint_incidence_vc2_20260921'
BASE = CODE.parent / 'results/joint_incidence_20260921'
SOURCE = BASE / 'source'
DATA = CODE.parent / 'results/joint_recall_iterations/explicit_context_capacity512'
SEEDS = [123, 456, 789]
TOPOLOGIES = [17]
LENGTHS = [64, 100, 772, 1540, 3076]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def bootstrap():
    sys.path[:0] = [str(CODE), '/u/archerdw/triton371', str(SOURCE/'deltaai'),
                   str(SOURCE/'smat'), str(SOURCE),
                   '/u/archerdw/.local/lib/python3.12/site-packages']
    os.environ.update(FLA_TILELANG='0', WANDB_MODE='disabled',
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      TRITON_F32_DEFAULT='tf32x3')
    import sitecustomize  # NumPy compatibility from the frozen source.


def verify_sources():
    path = ROOT/'source_sha256.json'
    for filename, expected in json.loads(path.read_text()).items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_data():
    manifest = json.loads((DATA/'data/manifest.json').read_text())
    for name, record in manifest['files'].items():
        if '-unique.npy' in name:
            continue
        digest = hashlib.file_digest((DATA/'data'/name).open('rb'), 'sha256').hexdigest()
        assert digest == record['sha256'], name
    return hashlib.sha256((DATA/'data/manifest.json').read_bytes()).hexdigest()

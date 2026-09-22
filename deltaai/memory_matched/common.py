"""Isolated state-memory-matched MQAR experiment; no changes to paper runs."""
import hashlib
import json
import os
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent / 'results/mqar_memory_matched_20260920'
SOURCE = ROOT / 'source'
SEEDS = [123, 1, 2, 3, 4]
STATE = 736


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def bootstrap():
    os.environ.update(MQAR_FAMILY='mamba2', ZOO_DM='16', ZOO_DS='1',
                      ZOO_EPOCHS='32', WANDB_MODE='disabled', FLA_TILELANG='0',
                      TRITON_F32_DEFAULT='tf32x3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    os.environ.pop('ZOO_SMOKE', None)
    sys.path[:0] = [str(CODE), '/u/archerdw/triton371', str(SOURCE/'deltaai'),
                   str(SOURCE/'smat'), str(SOURCE),
                   '/u/archerdw/.local/lib/python3.12/site-packages']


def verify_sources():
    manifest = json.loads((ROOT/'source_sha256.json').read_text())
    for path, expected in manifest.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    return hashlib.sha256((ROOT/'source_sha256.json').read_bytes()).hexdigest()


def make_config(seed, folder):
    os.environ['MQAR_RUN_DIR'] = str(folder)
    from zoo_mqar_four_reads_sweep_configs import configs
    config = configs[0].model_copy(deep=True)
    config.seed = seed
    config.learning_rate = .01
    config.max_epochs = 32
    config.early_stopping_metric = None
    config.run_id = config.model.name = f'mamba2-w16-state{STATE}-s{seed}'
    config.sweep_id = 'mqar-state-memory-matched-20260920'
    config.model.sequence_mixer.name = 'model.StateMatchedMamba2'
    config.model.sequence_mixer.kwargs = dict(d_state=STATE)
    return config

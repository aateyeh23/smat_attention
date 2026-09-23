"""Paths and fixed protocol for the one-job Mamba shared-feature pilot."""
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path('/u/archerdw/smat_attention')
ROOT = REPO / 'deltaai/results/learned_mkar_original_mamba_transport_normalized_stop_20260923'
SOURCE = ROOT / 'source'
SEEDS = [123]
KS = [1, 2, 3]
CONFIGS = [dict(name='mamba2-d4-normalized-route-stop',family='mamba2',d=4,random_incidence=False,detach_write_hash_input=True,memory_boundary_transport=True,memory_incidence_rescale=True,detach_write_hash_features=True)]
PROTOCOL = dict(length=1024, boundary=512, records=8, answers=8, vocab=290,
    requested_sizes=KS, width=256, layers=1, ffn_width=1024,
    head_dim=64, state_dim=64, gdn_heads=4, softmax_heads=4,
    batch=32, steps=12000, epochs=32, steps_per_epoch=375,
    learning_rate=.0003, weight_decay=.1, gradient_clip=1.,
    optimizer='AdamW on all parameters, matching original task script',
    scheduler='constant; no warmup', precision='FP32, matching original executable task',
    eval_every=2000, eval_batches=8, seed_list=SEEDS,
    scoring='Original evaluate(): raw output > 0.5; exact binary support and MSE',
    data='Original MKAR.batch unchanged: fresh noisy sequences, separate key/payload '
        'tokens, basis-vector payloads, eight serialized queries per sequence; '
        'a separate model is trained for each fixed k. Evaluation advances the same '
        'generator exactly as in the original script.',
    loss='Original binary_cross_entropy_with_logits on answer rows; learned SMat '
        'retains its existing auxiliary routing balance loss (coefficient 0.01).',
    model='Original MKARModel interface and common Block including learned positions, '
        'payload injection, LayerNorm, residual conv3 and 4x GELU FFN; only attn mixer '
        'replaced for native/learned GDN or Mamba2. Softmax uses original Attention.',
    smat='Learned one-write/four-read routing; no oracle directions or planted addresses; '
        'Mamba independent writes with normalized shared query/key projection and causal writer features; neighboring-cell writer gradient retained.',
    randomization='Unrestricted Curveball, 100 times number of summaries, exact margins',
    early_stopping=False, checkpoint_selection='final step', tuning='none')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bootstrap():
    sys.path[:0] = [str(SOURCE/'experiment'), str(SOURCE/'original/src'),
        str(SOURCE/'original/tasks'), '/u/archerdw/triton371',
        str(SOURCE/'deltaai'), str(SOURCE/'smat'), str(SOURCE),
        '/u/archerdw/.local/lib/python3.12/site-packages']
    os.environ.update(FLA_TILELANG='0', WANDB_MODE='disabled',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TRITON_F32_DEFAULT='tf32x3')
    import sitecustomize  # noqa: F401


def verify_sources():
    path = ROOT/'source_sha256.json'
    for relative, expected in json.loads(path.read_text()).items():
        assert digest(SOURCE/relative) == expected, relative
    assert json.loads((ROOT/'protocol.json').read_text()) == PROTOCOL
    return digest(path)

"""Paths and fixed protocol for the learned Mamba-2 + SMat multi-key recall runs.

The campaign ran from a frozen copy of the modules it imports.  Every one of
those is byte-identical in this repository (src/smat, tasks/mkar.py,
src/smat_lm, third_party/zoology, experiments/sitecustomize.py) except the
scrubbed comment in src/smat_lm/zoo_smat_mixer.py, so without the frozen copy
the scripts run from the repository modules instead.  MKAR_CAMPAIGN picks the
seed list (pilot_s123: the selection seed; confirm: the four confirmation
seeds) and MKAR_ROOT the output directory."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAMPAIGNS = {'pilot_s123': [123], 'confirm': [456, 789, 2026, 2027]}
CAMPAIGN = os.environ.get('MKAR_CAMPAIGN', 'confirm')
ROOT = Path(os.environ.get('MKAR_ROOT', REPO / 'results/mkar_mamba_smat_20260923' / CAMPAIGN))
SOURCE = ROOT / 'source'
SEEDS = CAMPAIGNS[CAMPAIGN]
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
    if SOURCE.is_dir():
        # the frozen copy, as the campaign ran
        paths = [SOURCE/'experiment', SOURCE/'original/src', SOURCE/'original/tasks',
                 SOURCE/'code', SOURCE/'smat', SOURCE]
    else:
        paths = [HERE, REPO/'src', REPO/'tasks', REPO/'src/smat_lm',
                 REPO/'third_party', REPO/'experiments']
    sys.path[:0] = [str(p) for p in paths]
    os.environ.update(FLA_TILELANG='0', WANDB_MODE='disabled',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TRITON_F32_DEFAULT='tf32x3')
    import sitecustomize  # noqa: F401


# where each entry of source_sha256.json lives in this repository
_REPO_PLACES = {'experiment/': HERE, 'original/src/': REPO/'src', 'original/tasks/': REPO/'tasks',
    'code/': REPO/'src/smat_lm', 'smat/': REPO/'src/smat_lm', 'zoology/': REPO/'third_party/zoology'}
# src/smat/__init__.py has only a docstring, whose module list changed when
# modules the paper does not use were removed.
_DOCSTRING_ONLY = {'original/src/smat/__init__.py'}


def verify_sources():
    path = ROOT/'source_sha256.json'
    manifest = json.loads(path.read_text())
    if SOURCE.is_dir():
        for relative, expected in manifest.items():
            assert digest(SOURCE/relative) == expected, relative
    else:
        # Files edited for this repository are listed with every SHA-256 they
        # ran under in the SCRUBBED.json beside them; anything else must match.
        known = {}
        for place in (HERE, REPO/'src/smat_lm'):
            if (place/'SCRUBBED.json').exists():
                for name, entry in json.loads((place/'SCRUBBED.json').read_text())['files'].items():
                    known[str(place/name)] = {v for k, v in entry.items() if k.startswith('sha256')}
        for relative, expected in manifest.items():
            prefix = next((p for p in _REPO_PLACES if relative.startswith(p)), None)
            if prefix is None or relative in _DOCSTRING_ONLY:
                continue
            candidate = _REPO_PLACES[prefix]/relative[len(prefix):]
            if not candidate.exists():
                continue   # part of the frozen working copy the model does not import
            if digest(candidate) != expected:
                assert expected in known.get(str(candidate), ()), relative
    assert json.loads((ROOT/'protocol.json').read_text()) == PROTOCOL
    return digest(path)

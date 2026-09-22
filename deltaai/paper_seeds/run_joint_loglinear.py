"""Use the frozen joint-recall trainer with a Log-Linear mixer."""
import joint_recall
import hashlib
import json
import os
import fcntl
import sys
from pathlib import Path

task=json.loads(os.environ['PAPER_TASK'])
gate_path=Path(__file__).resolve().parent.parent/'results/paper_seeds_20260920/loglinear_validated.json'
gate=json.loads(gate_path.read_text())
source_hash=hashlib.sha256(Path(__file__).with_name('joint_loglinear.py').read_bytes()).hexdigest()
assert gate['sha256']==source_hash and len(gate['model_checks'])==2
folder=Path(task['result']).parent
folder.mkdir(parents=True,exist_ok=True)
run_lock=(folder/'run.lock').open('a')
try:
    fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    print('RUN_ALREADY_OWNED '+task['name'],flush=True)
    sys.exit(0)
recipe=dict(family=task['family'],width=64,seed=task['seed'],levels=13,
    operator='base-2 WEAK Log-Linear',backend='dense PyTorch with batchwise activation checkpointing',
    upstream_commit='7f8644159c1406fae1ad863829a5b3a4fbf63022',
    logical_head_and_state_dim=16,time_padding=None,
    precision='BF16 autocast; dense operator and q/k normalization in FP32',
    source_sha256=source_hash)
path=folder/'loglinear_recipe.json'
if path.exists():
    assert json.loads(path.read_text())==recipe
else:
    path.write_text(json.dumps(recipe,indent=2)+'\n')


def make_model(family, width, d, lengths, vocab):
    from zoology.config import ModelConfig
    from zoology.model import LanguageModel
    import zoology.mixers.mamba2 as zm
    from zoo_smat_mixer import SmatMamba2Block
    zm.Mamba2Block = SmatMamba2Block
    assert d == 0 and max(lengths) <= 4096
    config = ModelConfig(block_type='Mamba2Block', d_model=width, n_layers=2,
        sequence_mixer=dict(name='joint_loglinear.JointLogLinearMixer', kwargs=dict(family=family)),
        max_position_embeddings=0, vocab_size=vocab, name='loglinear-'+family, embed_dropout=.1)
    return LanguageModel(config).cuda()


joint_recall.make_model = make_model
if __name__ == '__main__':
    joint_recall.main()

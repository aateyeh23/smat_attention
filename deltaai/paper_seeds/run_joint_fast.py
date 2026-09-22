"""Fresh fast-backend runs using the unchanged frozen joint-recall trainer."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import joint_recall
import joint_loglinear

task=json.loads(os.environ['PAPER_TASK'])
root=Path(task['result']).parents[2]
folder=Path(task['result']).parent
folder.mkdir(parents=True,exist_ok=True)
lock=(folder/'run.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    print('RUN_ALREADY_OWNED',task['name'],flush=True);sys.exit(0)
if task['family']=='mamba2':
    from upstream_adapter import log_linear
else:
    from tree64_operator import log_linear
joint_loglinear.log_linear=log_linear
recipe=dict(backend=task['backend'],seed=task['seed'],family=task['family'],levels=13,
    initialization='fresh from seed',source_manifest_sha256=hashlib.sha256((root/'source_sha256.json').read_bytes()).hexdigest(),
    original_campaign='paper_seeds_20260920',upstream_commit='7f8644159c1406fae1ad863829a5b3a4fbf63022')
path=folder/'fast_backend_recipe.json'
if path.exists():assert json.loads(path.read_text())==recipe
else:path.write_text(json.dumps(recipe,indent=2)+'\n')

def make_model(family,width,d,lengths,vocab):
    from zoology.config import ModelConfig
    from zoology.model import LanguageModel
    import zoology.mixers.mamba2 as zm
    from zoo_smat_mixer import SmatMamba2Block
    zm.Mamba2Block=SmatMamba2Block
    assert family==task['family'] and d==0 and max(lengths)<=4096
    config=ModelConfig(block_type='Mamba2Block',d_model=width,n_layers=2,
        sequence_mixer=dict(name='joint_loglinear.JointLogLinearMixer',kwargs=dict(family=family)),
        max_position_embeddings=0,vocab_size=vocab,name='loglinear-'+family,embed_dropout=.1)
    return LanguageModel(config).cuda()

joint_recall.make_model=make_model
if __name__=='__main__':joint_recall.main()

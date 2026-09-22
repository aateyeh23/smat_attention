"""Six one-seed Log-Linear MQAR arms, no baseline sweep."""
import hashlib
import json
import os
from pathlib import Path
import zoology.train as training
import zoology.mixers.mamba2 as zm
from zoology.config import TrainConfig, ModelConfig, LoggerConfig
try:                       # the second cluster has no mamba_ssm/causal_conv1d, which zoo_smat_mixer
    from zoo_smat_mixer import SmatMamba2Block      # imports at module scope; the
except ImportError:                                 # block itself needs only Zoology.
    from zoo_log_linear_block import SmatMamba2Block
from zoo_mqar_interleave import InterleavedTrainer
from zoo_reference_configs import data, DM, EPOCHS, VOCAB_SIZE
import zoo_log_linear

class LogLinearTrainer(InterleavedTrainer):
    def fit(self):
        self.early_stopping_metric = None
        return super().fit()

training.Trainer = LogLinearTrainer
zm.Mamba2Block = SmatMamba2Block
family = os.environ['LL_FAMILY']
lr = .003 if family == 'gdn' and DM == 64 else .01
source = Path(zoo_log_linear.__file__).resolve()
assert str(source) == os.environ.get('LL_SOURCE_PATH', '/opt/log-linear/zoo_log_linear.py'), source
assert hashlib.sha256(source.read_bytes()).hexdigest() == os.environ['LL_SOURCE_SHA256']
print('SOURCE_AUDIT ' + json.dumps(dict(path=str(source),sha256=os.environ['LL_SOURCE_SHA256'],
    family=family,width=DM,lr=lr,seed=123,backend='dense-pytorch',levels=9)),flush=True)
name = f'log_linear_{family}_w{DM}'
model = ModelConfig(block_type='Mamba2Block',d_model=DM,n_layers=2,
    sequence_mixer=dict(name='zoo_log_linear.LogLinearMixer',kwargs=dict(family=family)),
    max_position_embeddings=0,vocab_size=VOCAB_SIZE,name=name)
configs = [TrainConfig(model=model,data=data,learning_rate=lr,max_epochs=EPOCHS,
    seed=123,logger=LoggerConfig(project_name='smat-zoology'),slice_keys=['num_kv_pairs'],
    sweep_id='log-linear-s123',run_id=name)]

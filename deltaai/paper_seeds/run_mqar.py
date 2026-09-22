"""Repeat a frozen Table 2 configuration, changing only the model seed."""
import json
import os
from pathlib import Path
import sys

os.environ['TRITON_CACHE_DIR'] = f'/tmp/{os.environ["USER"]}-paper-seeds-triton'
task = json.loads(os.environ['PAPER_TASK'])
if task['d'] == 0 and not (Path(__file__).resolve().parent.parent/'results/paper_seeds_20260920/mqar_loglinear_recipe_verified.json').exists():
    print('WAITING_FOR_TABLE2_LOGLINEAR_PROVENANCE', flush=True)
    sys.exit(0)

import zoology.train as training
from zoo_mqar_four_reads_sweep_configs import configs
from zoo_mqar_interleave import InterleavedTrainer


class FullBudgetTrainer(InterleavedTrainer):
    def fit(self):
        self.early_stopping_metric = None
        return super().fit()


training.Trainer = FullBudgetTrainer
config, = configs
config.seed = task['seed']
config.learning_rate = task['lr']
config.run_id = config.model.name = task['name']
config.sweep_id = 'paper-five-seeds-20260920'
config.early_stopping_metric = None
if task['d'] == 0:
    config.model.sequence_mixer.name = 'zoo_log_linear.LogLinearMixer'
    config.model.sequence_mixer.kwargs = dict(family=task['family'])
elif task['family'] == 'gdn' and task['width'] >= 32 and task['d'] >= 2:
    config.model.sequence_mixer.name = 'zoo_gdn_transport.SmatGDNTransport'
    config.model.sequence_mixer.kwargs.update(
        write_hash_neighbor_grad=True, transport_scalar_decay=True,
        detach_write_hash_input=True, memory_plant=False,
        memory_incidence_rescale=False)
folder = Path(os.environ['MQAR_RUN_DIR'])
folder.mkdir(parents=True, exist_ok=True)
(folder/'config.json').write_text(config.model_dump_json(indent=2)+'\n')
training.train(config)
status = json.loads((folder/f"w{task['width']}-d{os.environ['ZOO_DS']}.json").read_text())
if status['complete']:
    assert status['next_epoch'] == 32
    (folder/'result.json').write_text(json.dumps(dict(task, **status), indent=2)+'\n')

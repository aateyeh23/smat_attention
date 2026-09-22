"""Task-general candidates, same width/data/seed/budget and strict epoch gates."""
import json
import os
from zoo_gdn_transport_configs import configs
import zoology.train as training
from zoo_gdn_gated_trainer import BaselineGatedTrainer

training.Trainer = BaselineGatedTrainer
variant = json.loads(os.environ['GDN_VARIANT_JSON'])
for config in configs:
    config.model.sequence_mixer.kwargs.update(variant['kwargs'])
    config.model.name = 'gdn_smat_w32_d3_'+variant['name']
    config.run_id = config.model.name

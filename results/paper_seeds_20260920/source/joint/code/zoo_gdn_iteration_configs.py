"""Task-general candidates, same width/data/seed/budget and strict epoch gates."""
import json
import os
variant = json.loads(os.environ['GDN_VARIANT_JSON'])
if variant.get('baseline', False):
    from zoo_mqar_four_reads_sweep_configs import configs
else:
    from zoo_gdn_transport_configs import configs
import zoology.train as training
from zoo_gdn_gated_trainer import BaselineGatedTrainer

training.Trainer = BaselineGatedTrainer
for config in configs:
    # Named historical trials keep their original defaults when the current
    # standalone recipe changes. Each trial then applies its explicit overrides.
    if not variant.get('baseline', False):
        config.model.sequence_mixer.kwargs.update(
            write_hash_neighbor_grad=False, transport_scalar_decay=True,
            detach_write_hash_input=False, memory_plant=True,
            memory_incidence_rescale=False,
        )
    config.model.sequence_mixer.kwargs.update(variant['kwargs'])
    width = config.model.d_model
    d = config.model.sequence_mixer.kwargs['d']
    if width != int(os.environ['ZOO_DM']) or d != variant.get('d', 3):
        raise ValueError('Loaded configuration does not match the requested width/d')
    config.model.name = f'gdn_smat_w{width}_d{d}_'+variant['name']
    config.learning_rate = variant.get('lr', config.learning_rate)
    config.run_id = config.model.name

"""Task-general candidates, same width/data/seed/budget and strict epoch gates."""
import json
import os
from zoo_gdn_transport_configs import configs
import zoology.train as training
from zoo_gdn_gated_trainer import BaselineGatedTrainer

training.Trainer = BaselineGatedTrainer
variant = json.loads(os.environ['GDN_VARIANT_JSON'])
for config in configs:
    # Named historical trials keep their original defaults when the current
    # standalone recipe changes. Each trial then applies its explicit overrides.
    config.model.sequence_mixer.kwargs.update(
        write_hash_neighbor_grad=False, transport_scalar_decay=True,
        detach_write_hash_input=False, memory_plant=True,
        memory_incidence_rescale=False,
    )
    config.model.sequence_mixer.kwargs.update(variant['kwargs'])
    width = config.model.d_model
    d = config.model.sequence_mixer.kwargs['d']
    if width != 32 or d != variant.get('d', 3):
        raise ValueError('Loaded configuration does not match the requested width/d')
    config.model.name = f'gdn_smat_w{width}_d{d}_'+variant['name']
    config.run_id = config.model.name

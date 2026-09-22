"""Opt-in transport variant: one forward write, neighboring-cell backward."""
from zoo_gdn_transport_configs import configs

for config in configs:
    config.model.sequence_mixer.kwargs['write_hash_neighbor_grad'] = True
    config.model.name += '_neighbor_write_grad'
    config.run_id = config.model.name

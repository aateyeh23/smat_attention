"""Compatibility entrypoint; neighboring write gradients are now in the recipe."""
from zoo_gdn_transport_configs import configs

for config in configs:
    config.model.sequence_mixer.kwargs['write_hash_neighbor_grad'] = True
    config.model.name += '_neighbor_write_grad'
    config.run_id = config.model.name

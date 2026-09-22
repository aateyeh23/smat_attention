"""Tie memory key/query features and write each value under its preceding key."""
from zoo_gdn_aligned_configs import configs
for config in configs:
    config.model.sequence_mixer.kwargs['memory_key_mode'] = 'tied_previous'
    config.model.name += '_tied_memory_keys'
    config.run_id += '_tied_memory_keys'

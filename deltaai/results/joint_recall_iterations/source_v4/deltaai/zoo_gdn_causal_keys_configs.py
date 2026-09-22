"""Learn causal write-key context; share the memory feature projection with queries."""
from zoo_gdn_delta_configs import configs
for config in configs:
    config.model.sequence_mixer.kwargs['memory_key_mode'] = 'tied_causal'
    config.model.name += '_causal_memory_keys'
    config.run_id += '_causal_memory_keys'

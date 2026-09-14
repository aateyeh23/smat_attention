"""Keep the sparse neighboring-cell routing mixture soft throughout train and eval."""
from zoo_gdn_causal_keys_configs import configs
for config in configs:
    config.model.sequence_mixer.kwargs['memory_route_soft'] = True
    config.model.name += '_soft_routes'
    config.run_id += '_soft_routes'

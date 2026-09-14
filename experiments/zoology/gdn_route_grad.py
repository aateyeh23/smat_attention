"""Restore neighbor-cell query gradients while keeping the hard forward routing."""
from zoo_gdn_causal_keys_configs import configs
for config in configs:
    config.model.sequence_mixer.kwargs['memory_route_full_grad'] = True
    config.model.name += '_full_route_grad'
    config.run_id += '_full_route_grad'

"""Keep preceding-key routing soft for eight epochs before hard assignments."""
from zoo_gdn_aligned_configs import configs
for config in configs:
    config.model.sequence_mixer.kwargs['anneal_steps'] = 8 * 707
    config.model.name += '_anneal8ep'
    config.run_id += '_anneal8ep'

"""Learn causal memory context with a slower transition to hard routing."""
from zoo_gdn_causal_keys_configs import configs
for config in configs:
    config.model.sequence_mixer.kwargs['anneal_steps'] = 8 * 707
    config.model.name += '_anneal8ep'
    config.run_id += '_anneal8ep'

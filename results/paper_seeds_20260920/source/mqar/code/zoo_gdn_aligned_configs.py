"""MQAR routing ablation: hash the preceding key when writing its value."""
from zoo_gdn_delta_configs import configs

for config in configs:
    config.model.sequence_mixer.kwargs['hash_key_shift'] = 1
    config.model.name += '_prevkey'
    config.run_id += '_prevkey'

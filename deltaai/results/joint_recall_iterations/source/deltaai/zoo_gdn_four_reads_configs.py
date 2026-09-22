"""Width-16 MQAR: one write bucket, four distinct learned summary reads."""
from zoo_gdn_causal_keys_configs import configs

for config in configs:
    if config.model.d_model != 16:
        raise ValueError("This experiment requires model width 16")
    kwargs = config.model.sequence_mixer.kwargs
    if kwargs['d'] not in (2, 3, 4):
        raise ValueError("This experiment requires d=2,3,4")
    kwargs.update(memory_read_k=4, headdim=16, n_heads=1, expand_v=1)
    config.model.name += '_write1_read4'
    config.run_id += '_write1_read4'

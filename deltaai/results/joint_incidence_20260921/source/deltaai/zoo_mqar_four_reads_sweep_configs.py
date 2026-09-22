"""Matched seed-123 MQAR sweeps for GDN and Mamba2 with four-read SMAT."""
import os
from zoo_gdn_causal_keys_configs import configs

family = os.environ['MQAR_FAMILY']
if family not in ('gdn', 'mamba2'):
    raise ValueError(family)
for config in configs:
    width = config.model.d_model
    d = config.model.sequence_mixer.kwargs['d']
    if family == 'gdn':
        config.model.sequence_mixer.kwargs.update(
            memory_read_k=4 if d >= 2 else None,
            headdim=16, n_heads=1 if width == 16 else 2, expand_v=1)
    else:
        config.model.sequence_mixer.name = 'zoo_mamba_four_reads.MqarMambaFourReads'
        config.model.sequence_mixer.kwargs = dict(d=d)
    name = f'{family}_w{width}_d{d}_hd16_state16'
    if d >= 2:
        name += '_write1_read4'
    config.model.name = name
    config.run_id = name

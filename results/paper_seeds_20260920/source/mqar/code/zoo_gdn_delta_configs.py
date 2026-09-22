"""Matched single-seed MQAR: replace additive SMAT writes with delta updates."""
import os
os.environ['TRITON_F32_DEFAULT'] = 'tf32x3'
from zoo_gdn_interleaved_configs import configs

for config in configs:
    config.model.sequence_mixer.kwargs['memory_update'] = 'delta'
    config.model.name += '_delta'
    config.run_id += '_delta'

"""Same seed/data/optimizer recipe as the completed w32/d3 four-read run."""
import os
os.environ.update(MQAR_FAMILY='gdn',ZOO_DM='32',ZOO_DS='3',ZOO_EPOCHS='32')
from zoo_mqar_four_reads_sweep_configs import configs

for config in configs:
    config.model.sequence_mixer.name='zoo_gdn_transport.SmatGDNTransport'
    config.model.name='gdn_transport_w32_d3_hd16_state16_write1_read4'
    config.run_id=config.model.name

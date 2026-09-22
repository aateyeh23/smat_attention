"""Current transport recipe: learned scalar decay, no output rescaling."""
import os
os.environ['MQAR_FAMILY'] = 'gdn'
for name, value in dict(ZOO_DM='32', ZOO_DS='3', ZOO_EPOCHS='32').items():
    os.environ.setdefault(name, value)
from zoo_mqar_four_reads_sweep_configs import configs

DEFAULT_TRANSPORT_KWARGS = dict(
    write_hash_neighbor_grad=True,
    transport_scalar_decay=True,
    detach_write_hash_input=True,
    memory_plant=False,
    memory_incidence_rescale=False,
)

for config in configs:
    config.model.sequence_mixer.name='zoo_gdn_transport.SmatGDNTransport'
    config.model.sequence_mixer.kwargs.update(DEFAULT_TRANSPORT_KWARGS)
    width = config.model.d_model
    d = config.model.sequence_mixer.kwargs['d']
    config.model.name=f'gdn_transport_w{width}_d{d}_hd16_state16_write1_read4_decay_no_rescale'
    config.run_id=config.model.name

"""Native Mamba-2 with enlarged state; same wrapper as the frozen controls."""
from zoo_smat_mixer import SmatMamba2MR


class StateMatchedMamba2(SmatMamba2MR):
    def __init__(self, d_model, layer_idx=0, d_state=736, **kwargs):
        super().__init__(d_model, layer_idx=layer_idx, d=1, d_state=d_state,
            headdim=16, reset=True, read='chash', hash_codim=1,
            hash_src='hidden', hash_conv=True, anneal_steps=0, balance_coef=.01,
            sparse_ops=True, lam_act='sigmoid', lam_bias=-2.197224577,
            g_decay=False, g_write_mode='independent', g_write_init=.1, **kwargs)
        # d=1 has no profile memory, learned routing, or boundary reset.
        assert self.d == 1 and len(self.ca) == 0


def state_elements(mixer):
    m = mixer.mixer
    return dict(recurrent=m.nheads*m.headdim*m.d_state,
                convolution=m.conv1d.in_channels*m.d_conv)

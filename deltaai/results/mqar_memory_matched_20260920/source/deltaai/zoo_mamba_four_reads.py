"""Mamba2 MQAR control and learned one-write/four-read SMAT variant."""
import torch
from zoo_smat_mixer import SmatMamba2MR


class MqarMambaFourReads(SmatMamba2MR):
    def __init__(self, d_model, layer_idx=0, d=3, **kwargs):
        super().__init__(d_model, layer_idx=layer_idx, d=d, d_state=16,
            headdim=16, reset=True, read='chash', hash_codim=1,
            hash_src='hidden', hash_conv=True, anneal_steps=0, balance_coef=0.01,
            sparse_ops=True, lam_act='sigmoid', lam_bias=-2.197224577,
            g_decay=False, g_write_mode='independent', g_write_init=0.1, **kwargs)
        if d >= 2:
            for length in (64, 128, 256):
                self._spec(length, torch.device('cpu'))
                if int(self.specs[length].dim) != d-1:
                    raise ValueError('Requested geometry was reduced')
                for ca in self.ca[str(length)].values():
                    ca.enable_topk_reads(4, d_model)
                    ca.anneal = 1.0
                    ca.hard_k1 = True
            self.specs.clear()
        print(f'MQAR_MAMBA_FOUR_READS width={d_model} d={d} heads={self.h} '
              f'head_dim={self.p} state_dim={self.N} write=1 read=4', flush=True)

"""Plain recurrent backbones with the same embedding, FFN and loss as SMAT arms."""
from dataclasses import dataclass

import torch
from torch import nn

from gdn_smat_scale import ScaleConfig as OriginalConfig, ScaleLM as OriginalLM, RMSNorm
from gdn_smat_transport_scale import Block as SharedBlock
from zoo_smat_gdn import SmatGDNReset
from zoology.mixers.mamba2 import Mamba2


@dataclass
class ScaleConfig(OriginalConfig):
    family: str = 'gdn'
    memory_variant: str = 'plain_full_sequence'


class PlainMamba(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        if cfg.heads * cfg.head_dim != 2 * cfg.width:
            raise ValueError('Plain Mamba-2 must retain native expansion two')
        self.mixer = Mamba2(d_model=cfg.width, d_state=cfg.head_dim, d_conv=4,
                            expand=2, headdim=cfg.head_dim, ngroups=1,
                            chunk_size=64, use_mem_eff_path=False, layer_idx=index)

    def forward(self, x):
        return self.mixer(x)

    def get_auxiliary_loss(self):
        return 0.


class Block(SharedBlock):
    def __init__(self, cfg, index):
        nn.Module.__init__(self)
        self.norm = RMSNorm(cfg.width, cfg.fused_norm)
        if cfg.family == 'gdn':
            self.mixer = SmatGDNReset(cfg.width, layer_idx=index, d=1,
                headdim=cfg.head_dim, n_heads=cfg.heads, expand_v=1, reset=False)
        elif cfg.family == 'mamba2':
            self.mixer = PlainMamba(cfg, index)
        else:
            raise ValueError(cfg.family)
        self.norm2 = RMSNorm(cfg.width, cfg.fused_norm)
        self.up_gate = nn.Linear(cfg.width, 2*cfg.ffn_width, bias=False)
        self.down = nn.Linear(cfg.ffn_width, cfg.width, bias=False)


class ScaleLM(OriginalLM):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        if cfg.d != 1:
            raise ValueError('Plain baselines use d=1 as the no-SMAT identifier')
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width, cfg.fused_norm)
        nn.init.normal_(self.embedding.weight, std=.02)

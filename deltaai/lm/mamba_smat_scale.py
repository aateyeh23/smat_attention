"""Mamba-2/SMAT PG19 model with corrected hard-write gradients."""
from dataclasses import dataclass
import torch
from torch import nn
from gdn_smat_scale import ScaleConfig as OriginalConfig, RMSNorm
from gdn_smat_transport_scale import Block as OriginalBlock, ScaleLM as OriginalLM
from zoo_smat_mixer import SmatMamba2MR


@dataclass
class ScaleConfig(OriginalConfig):
    memory_variant: str = 'mamba2_fixed_write_gradient_decay'
    read_rank: int | None = None


class MambaMixer(SmatMamba2MR):
    def __init__(self,cfg,index):
        super().__init__(cfg.width,layer_idx=index,d=cfg.d,d_state=cfg.head_dim,
            headdim=cfg.head_dim,mamba_heads=None if cfg.heads*cfg.head_dim==2*cfg.width else cfg.heads,reset=True,read='chash',hash_codim=1,
            hash_src='hidden',hash_conv=True,anneal_steps=0,balance_coef=.01,
            sparse_ops=True,lam_act='sigmoid',lam_bias=-2.197224577,
            g_decay=True,g_write_mode='independent',g_write_init=.1)
        self.detach_write_hash_input=True
        self._spec(cfg.length,torch.device('cpu'))
        assert self.specs[cfg.length].dim==cfg.d-1
        for modules in self.ca.values():
            for ca in modules.values():
                ca.enable_topk_reads(4,cfg.width,rank=cfg.read_rank)
                ca.anneal=1.;ca.hard_k1=True;ca.plant=False
                ca.write_hash_neighbor_grad=True
                ca.detach_balance_weights=True
                ca.read_backend=cfg.read_backend
        self.specs.clear()


class Block(OriginalBlock):
    def __init__(self,cfg,index):
        nn.Module.__init__(self)
        self.norm=RMSNorm(cfg.width,cfg.fused_norm)
        self.mixer=MambaMixer(cfg,index)
        self.norm2=RMSNorm(cfg.width,cfg.fused_norm)
        self.up_gate=nn.Linear(cfg.width,2*cfg.ffn_width,bias=False)
        self.down=nn.Linear(cfg.ffn_width,cfg.width,bias=False)


class ScaleLM(OriginalLM):
    def __init__(self,cfg):
        nn.Module.__init__(self)
        assert cfg.d in (3,4)
        self.cfg=cfg
        self.embedding=nn.Embedding(cfg.vocab,cfg.width)
        self.blocks=nn.ModuleList([Block(cfg,i) for i in range(cfg.layers)])
        self.norm=RMSNorm(cfg.width,cfg.fused_norm)
        nn.init.normal_(self.embedding.weight,std=.02)

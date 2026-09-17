"""PG-19 LM with the verified updated MQAR GDN/SMAT transport configuration."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from gdn_smat_scale import ScaleConfig as OriginalConfig, ScaleLM as OriginalLM, RMSNorm
from zoo_gdn_transport import SmatGDNTransport


@dataclass
class ScaleConfig(OriginalConfig):
    memory_variant: str = 'updated_transport_decay_no_rescale'
    read_rank: int | None = None


class Block(nn.Module):
    def __init__(self,cfg,index):
        super().__init__()
        self.norm=RMSNorm(cfg.width,cfg.fused_norm)
        self.mixer=SmatGDNTransport(cfg.width,layer_idx=index,d=cfg.d,
            headdim=cfg.head_dim,n_heads=cfg.heads,expand_v=1,reset=True,
            prebuild_lengths=(cfg.length,),memory_update='delta',
            memory_key_mode='tied_causal',memory_read_k=4,
            write_hash_neighbor_grad=True,transport_scalar_decay=True,
            detach_write_hash_input=True,memory_plant=False,memory_incidence_rescale=False)
        for modules in self.mixer.ca.values():
            for ca in modules.values():
                if cfg.read_rank is not None:
                    ca.enable_topk_reads(4,cfg.width,rank=cfg.read_rank)
                ca.read_backend=cfg.read_backend
        self.norm2=RMSNorm(cfg.width,cfg.fused_norm)
        self.up_gate=nn.Linear(cfg.width,2*cfg.ffn_width,bias=False)
        self.down=nn.Linear(cfg.ffn_width,cfg.width,bias=False)

    def forward(self,x):
        x=x+self.mixer(self.norm(x))
        aux=self.mixer.get_auxiliary_loss()
        if not torch.is_tensor(aux):aux=x.new_zeros((),dtype=torch.float32)
        gate,value=self.up_gate(self.norm2(x)).chunk(2,dim=-1)
        return x+self.down(F.silu(gate)*value),aux


class ScaleLM(OriginalLM):
    def __init__(self,cfg):
        nn.Module.__init__(self)
        if cfg.d not in (3,4):raise ValueError('This campaign contains only d=3 and d=4')
        self.cfg=cfg
        self.embedding=nn.Embedding(cfg.vocab,cfg.width)
        self.blocks=nn.ModuleList([Block(cfg,i) for i in range(cfg.layers)])
        self.norm=RMSNorm(cfg.width,cfg.fused_norm)
        nn.init.normal_(self.embedding.weight,std=.02)

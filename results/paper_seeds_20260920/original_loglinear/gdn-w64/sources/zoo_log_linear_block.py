"""SmatMamba2Block, lifted verbatim out of zoo_smat_mixer for the the second cluster runs.

zoo_smat_mixer imports mamba_ssm and causal_conv1d at module scope; neither is
installed on this cluster, and neither is reachable from the Log-Linear arms --
the only thing they take from that module is the block below, whose body uses
Zoology alone.  The class is copied character for character so the Log-Linear
arms get the same block, the same RMSNorm and the same init as every other
Zoology arm in this repo.
"""
import torch.nn as nn
from zoology.mixers.mamba2 import Mamba2Block as _ZooMamba2Block


class SmatMamba2Block(_ZooMamba2Block):
    """Zoology's Mamba2Block (RMSNorm, no MLP, add->norm->mixer, Mamba init path) with the
    SMAT mixer from config.sequence_mixer in place of the vendored Mamba2.  Installed by
    zoo_smat_configs via `zoology.mixers.mamba2.Mamba2Block = SmatMamba2Block`, so
    block_type="Mamba2Block" gives SMAT arms exactly the reference arm's block and init."""
    def __init__(self, config, layer_idx=0, **kwargs):
        nn.Module.__init__(self)
        from zoology.mixers.mamba2 import RMSNorm
        self.residual_in_fp32, self.fused_add_norm = False, False
        self.norm = RMSNorm(config.d_model)
        self.mixer = config.sequence_mixer.instantiate(d_model=config.d_model, layer_idx=layer_idx)
        self.mlp = None

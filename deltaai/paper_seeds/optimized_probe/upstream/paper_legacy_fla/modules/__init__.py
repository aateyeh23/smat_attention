# -*- coding: utf-8 -*-

from paper_legacy_fla.modules.convolution import ImplicitLongConvolution, LongConvolution, ShortConvolution
from paper_legacy_fla.modules.fused_bitlinear import BitLinear, FusedBitLinear
from paper_legacy_fla.modules.fused_cross_entropy import FusedCrossEntropyLoss
from paper_legacy_fla.modules.fused_kl_div import FusedKLDivLoss
from paper_legacy_fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from paper_legacy_fla.modules.fused_norm_gate import (
    FusedLayerNormGated,
    FusedLayerNormSwishGate,
    FusedLayerNormSwishGateLinear,
    FusedRMSNormGated,
    FusedRMSNormSwishGate,
    FusedRMSNormSwishGateLinear
)
from paper_legacy_fla.modules.layernorm import GroupNorm, GroupNormLinear, LayerNorm, LayerNormLinear, RMSNorm, RMSNormLinear
from paper_legacy_fla.modules.mlp import GatedMLP
from paper_legacy_fla.modules.rotary import RotaryEmbedding

__all__ = [
    'ImplicitLongConvolution', 'LongConvolution', 'ShortConvolution',
    'BitLinear', 'FusedBitLinear',
    'FusedCrossEntropyLoss', 'FusedLinearCrossEntropyLoss', 'FusedKLDivLoss',
    'GroupNorm', 'GroupNormLinear', 'LayerNorm', 'LayerNormLinear', 'RMSNorm', 'RMSNormLinear',
    'FusedLayerNormGated', 'FusedLayerNormSwishGate', 'FusedLayerNormSwishGateLinear',
    'FusedRMSNormGated', 'FusedRMSNormSwishGate', 'FusedRMSNormSwishGateLinear',
    'GatedMLP',
    'RotaryEmbedding'
]

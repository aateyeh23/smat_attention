# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from paper_legacy_fla.models.bitnet.configuration_bitnet import BitNetConfig
from paper_legacy_fla.models.bitnet.modeling_bitnet import BitNetForCausalLM, BitNetModel

AutoConfig.register(BitNetConfig.model_type, BitNetConfig)
AutoModel.register(BitNetConfig, BitNetModel)
AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM)


__all__ = ['BitNetConfig', 'BitNetForCausalLM', 'BitNetModel']

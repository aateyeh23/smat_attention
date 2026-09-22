# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from paper_legacy_fla.models.lightnet.configuration_lightnet import LightNetConfig
from paper_legacy_fla.models.lightnet.modeling_lightnet import LightNetForCausalLM, LightNetModel

AutoConfig.register(LightNetConfig.model_type, LightNetConfig)
AutoModel.register(LightNetConfig, LightNetModel)
AutoModelForCausalLM.register(LightNetConfig, LightNetForCausalLM)


__all__ = ['LightNetConfig', 'LightNetForCausalLM', 'LightNetModel']

# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from paper_legacy_fla.models.retnet.configuration_retnet import RetNetConfig
from paper_legacy_fla.models.retnet.modeling_retnet import RetNetForCausalLM, RetNetModel

AutoConfig.register(RetNetConfig.model_type, RetNetConfig)
AutoModel.register(RetNetConfig, RetNetModel)
AutoModelForCausalLM.register(RetNetConfig, RetNetForCausalLM)


__all__ = ['RetNetConfig', 'RetNetForCausalLM', 'RetNetModel']

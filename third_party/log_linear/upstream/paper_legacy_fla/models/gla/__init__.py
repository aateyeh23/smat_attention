# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from paper_legacy_fla.models.gla.configuration_gla import GLAConfig
from paper_legacy_fla.models.gla.modeling_gla import GLAForCausalLM, GLAModel

AutoConfig.register(GLAConfig.model_type, GLAConfig)
AutoModel.register(GLAConfig, GLAModel)
AutoModelForCausalLM.register(GLAConfig, GLAForCausalLM)


__all__ = ['GLAConfig', 'GLAForCausalLM', 'GLAModel']

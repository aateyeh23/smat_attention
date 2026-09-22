# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from paper_legacy_fla.models.gsa.configuration_gsa import GSAConfig
from paper_legacy_fla.models.gsa.modeling_gsa import GSAForCausalLM, GSAModel

AutoConfig.register(GSAConfig.model_type, GSAConfig)
AutoModel.register(GSAConfig, GSAModel)
AutoModelForCausalLM.register(GSAConfig, GSAForCausalLM)


__all__ = ['GSAConfig', 'GSAForCausalLM', 'GSAModel']

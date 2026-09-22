# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from paper_legacy_fla.models.mamba.configuration_mamba import MambaConfig
from paper_legacy_fla.models.mamba.modeling_mamba import MambaBlock, MambaForCausalLM, MambaModel

AutoConfig.register(MambaConfig.model_type, MambaConfig, True)
AutoModel.register(MambaConfig, MambaModel, True)
AutoModelForCausalLM.register(MambaConfig, MambaForCausalLM, True)


__all__ = ['MambaConfig', 'MambaForCausalLM', 'MambaModel', 'MambaBlock']

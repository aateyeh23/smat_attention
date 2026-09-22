"""Single-seed width-16 MQAR: Mamba-2 control and SMAT/reset independent writes."""
import os
import uuid
import torch
from zoology.config import TrainConfig, ModelConfig, LoggerConfig
import zoology.mixers.mamba2 as zm
from zoo_smat_mixer import SmatMamba2Block, SmatMamba2MR
from zoo_reference_configs import data, DM, EPOCHS, VOCAB_SIZE


class MqarMambaReset(SmatMamba2MR):
    def __init__(self, d_model, layer_idx=0, d=3, **kwargs):
        super().__init__(d_model, layer_idx=layer_idx, d=d, d_state=16,
            headdim=16, reset=True, read="chash", hash_codim=1,
            hash_src="hidden", hash_conv=True, anneal_steps=1000, balance_coef=0.01,
            sparse_ops=True, lam_act="sigmoid", lam_bias=-2.197224577,
            g_decay=False, g_write_mode="independent", g_write_init=0.1, **kwargs)
        assert self.mixer.headdim == 16 and self.mixer.d_state == 16
        print(f"MQAR_MAMBA_CONFIG layer={layer_idx} d={d} width={d_model} "
              f"headdim={self.mixer.headdim} d_state={self.mixer.d_state} heads={self.mixer.nheads}", flush=True)
        if d >= 2:
            for length in (64, 128, 256):
                self._spec(length, torch.device("cpu"))
            self.specs.clear()


if os.environ.get("MQAR_RUN_DIR"):
    import zoology.train as training
    from zoo_mqar_resume import ResumableTrainer
    training.Trainer = ResumableTrainer
zm.Mamba2Block = SmatMamba2Block
configs = []
for d in map(int, os.environ.get("ZOO_DS", "1,2,3,4").split(",")):
    name = f"mamba2_ctrl_w{DM}" if d == 1 else f"smat_mamba2_reset_independent_d{d}_c1_w{DM}"
    name += "_hd16_ds16"
    model = ModelConfig(block_type="Mamba2Block", d_model=DM, n_layers=2,
        sequence_mixer=dict(name="zoo_mamba_reset_configs.MqarMambaReset", kwargs=dict(d=d)),
        max_position_embeddings=0, vocab_size=VOCAB_SIZE, name=name)
    configs.append(TrainConfig(model=model, data=data, learning_rate=1e-2,
        max_epochs=EPOCHS, seed=123, logger=LoggerConfig(project_name="smat-zoology"),
        slice_keys=["num_kv_pairs"], sweep_id="mamba-reset-independent-" + uuid.uuid4().hex[:6], run_id=name))

"""Single-seed sparse SMAT-GDN/reset MQAR, using the existing Zoology recipe."""
import os
import uuid
from zoology.config import TrainConfig, ModelConfig, LoggerConfig
import zoology.mixers.mamba2 as zm
from zoo_smat_mixer import SmatMamba2Block
from zoo_reference_configs import data, DM, EPOCHS, VOCAB_SIZE
if os.environ.get("MQAR_RUN_DIR"):
    import zoology.train as training
    from zoo_mqar_resume import ResumableTrainer
    training.Trainer = ResumableTrainer

zm.Mamba2Block = SmatMamba2Block
configs = []
for d in map(int, os.environ.get("ZOO_DS", "1,2,3,4").split(",")):
    name = f"gdn_ctrl_w{DM}" if d == 1 else f"smat_gdn_reset_d{d}_c1_w{DM}"
    model = ModelConfig(block_type="Mamba2Block", d_model=DM, n_layers=2,
        sequence_mixer=dict(name="zoo_smat_gdn.SmatGDNReset", kwargs=dict(
            d=d, reset=True, headdim=16, expand_v=1, lam_bias=-2.197224577,
            anneal_steps=1000, balance_coef=0.01, hash_codim=1)),
        max_position_embeddings=0, vocab_size=VOCAB_SIZE, name=name)
    configs.append(TrainConfig(model=model, data=data, learning_rate=1e-2,
        max_epochs=EPOCHS, seed=123, logger=LoggerConfig(project_name="smat-zoology"),
        slice_keys=["num_kv_pairs"], sweep_id="gdn-reset-" + uuid.uuid4().hex[:6], run_id=name))

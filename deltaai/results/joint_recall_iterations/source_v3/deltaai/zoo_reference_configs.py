"""Zoology MQAR reference sweep, following the repo's original_mqar_configs.py
(and hence the Based / Log-Linear-paper protocol): ONE model trained on the
mixture of five cells, tested per cell, lr swept over logspace(-3,-1.5,4),
32 epochs, early stop at valid accuracy 0.99.  Width (d_model) from the
environment variable ZOO_DM (default 16); models: attention (conv + MHA) and
Mamba-2.  Run:  python -m zoology.launch zoo_reference_configs.py
"""
import os, uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB_SIZE = 8_192
DM = int(os.environ.get("ZOO_DM", "16"))
EPOCHS = int(os.environ.get("ZOO_EPOCHS", "32"))
SMALL = os.environ.get("ZOO_SMOKE") == "1"
sweep_name = f"smat-zoo-ref-dm{DM}-" + uuid.uuid4().hex[:6]

k = 0.01 if SMALL else 1.0
train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=int(100_000 * k), num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=int(20_000 * k),  num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=int(20_000 * k),  num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=int(20_000 * k),  num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=int(20_000 * k),  num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
    # the 512/128 and 1024/256 length-generalisation cells are excluded, as in the Log-Linear paper
]
input_seq_len = max(c.input_seq_len for c in train_configs + test_configs)
batch_size = 256
data = DataConfig(train_configs=train_configs, test_configs=test_configs,
                  batch_size=(batch_size, batch_size // 8),
                  cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "zoology_cache"))

model_factory_kwargs = {"state_mixer": dict(name="torch.nn.Identity", kwargs={}), "vocab_size": VOCAB_SIZE}
conv_mixer = dict(name="zoology.mixers.base_conv.BaseConv",
                  kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True})

models = []
attn = ModuleConfig(name="zoology.mixers.hybrid.Hybrid",
                    kwargs={"configs": [conv_mixer, dict(name="zoology.mixers.attention.MHA",
                                                          kwargs={"dropout": 0.1, "num_heads": 2})]})
models.append(ModelConfig(block_type="TransformerBlock", d_model=DM, n_layers=2, sequence_mixer=attn,
                          max_position_embeddings=input_seq_len, name="attention", **model_factory_kwargs))
models.append(ModelConfig(block_type="Mamba2Block", d_model=DM, n_layers=2,
                          sequence_mixer=dict(name="zoology.mixers.mamba2.Mamba2", kwargs={"d_state": 128, "headdim": min(64, 2 * DM)}),
                          max_position_embeddings=0, name="mamba2", **model_factory_kwargs))

_sel = os.environ.get("ZOO_MODELS")                       # e.g. "mamba2" to run one arm only
if _sel:
    models = [m for m in models if m.name in _sel.split(",")]
configs = []
for model in models:
    for lr in ([1e-3] if SMALL else ([float(x) for x in os.environ["ZOO_LRS"].split(",")] if os.environ.get("ZOO_LRS") else np.logspace(-3, -1.5, 4))):
        configs.append(TrainConfig(model=model, data=data, learning_rate=float(lr), max_epochs=EPOCHS,
                                   logger=LoggerConfig(project_name="smat-zoology"),
                                   slice_keys=["num_kv_pairs"], sweep_id=sweep_name,
                                   run_id=f"{model.name}-dm{DM}-lr{lr:.1e}"))

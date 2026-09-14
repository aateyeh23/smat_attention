"""Matched GDN/SMAT-reset ablation: change only the order of training batches."""
from zoo_gdn_reset_configs import configs
import zoology.train as training
from zoo_mqar_interleave import InterleavedTrainer
training.Trainer = InterleavedTrainer
for config in configs:
    config.run_id += '_interleaved'

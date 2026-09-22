"""Shuffle whole MQAR batches across task cells, preserving each cell's weight."""
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Sampler
from zoo_mqar_resume import ResumableTrainer


class EpochBatchSampler(Sampler):
    def __init__(self, size, epoch, seed=123):
        self.size, self.epoch, self.seed = size, epoch, seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self):
        return self.size


class InterleavedTrainer(ResumableTrainer):
    def train_epoch(self, epoch_idx):
        original = self.train_dataloader
        sampler = EpochBatchSampler(len(original.dataset), epoch_idx)
        self.train_dataloader = DataLoader(original.dataset, batch_size=None,
                                          sampler=sampler, num_workers=0)
        print(f"INTERLEAVED epoch={epoch_idx} first_batch_indices={list(sampler)[:12]}", flush=True)
        try:
            return super().train_epoch(epoch_idx)
        finally:
            self.train_dataloader = original

    def test(self, epoch_idx):
        metrics = super().test(epoch_idx)
        folder = Path(os.environ['MQAR_RUN_DIR'])
        with (folder / f"w{os.environ['ZOO_DM']}-d{os.environ['ZOO_DS']}-history.jsonl").open('a') as f:
            f.write(json.dumps(dict(epoch=epoch_idx + 1, metrics=metrics), default=float) + '\n')
        return metrics

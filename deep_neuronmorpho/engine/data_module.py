"""Data module for loading and processing neuron data."""

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader


class NeuronDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.num_workers = cfg.training.num_workers
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage: str | None = None):
        self.train_dataset = instantiate(self.cfg.data.train_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

"""Data module for loading and processing neuron data."""

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader


class NeuronDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_sets = None
        self.val_sets = None
        self.test_sets = None
        self.num_workers = cfg.data.num_workers
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage: str | None = None):
        self.train_sets = instantiate(self.cfg.data.train_dataset)
        self.val_sets = [instantiate(dataset_cfg) for dataset_cfg in self.cfg.data.val_datasets]
        self.test_sets = [instantiate(dataset_cfg) for dataset_cfg in self.cfg.data.test_datasets]

    def train_dataloader(self):
        return DataLoader(
            self.train_sets,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.cfg.data.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
                drop_last=False,
            )
            for dataset in self.val_sets
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.cfg.data.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
                drop_last=False,
            )
            for dataset in self.test_sets
        ]

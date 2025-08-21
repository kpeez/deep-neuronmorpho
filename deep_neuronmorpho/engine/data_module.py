"""Data module for loading and processing neuron data."""

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from deep_neuronmorpho.data import ContrastiveNeuronDataset, NeuronDataset


class NeuronDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        transform_config: DictConfig,
        batch_size: int,
        num_workers: int,
        **kwargs,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.transform_config = transform_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.pin_memory = torch.cuda.is_available()
        self.kwargs = kwargs

    def setup(self, stage: str | None = None):
        if stage == "fit":
            base_dataset = NeuronDataset(self.dataset_root)
            self.train_dataset = ContrastiveNeuronDataset(
                base_dataset,
                transform=self.transform_config,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
            **self.kwargs,
        )

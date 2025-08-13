"""This module contains utilities for building dataloaders for the GraphDINO model."""

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, default_collate

from .dataset import GraphDINODataset


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def build_dataloader(cfg: DictConfig):
    num_workers = cfg.training.num_workers if cfg.training.num_workers is not None else 0
    kwargs = (
        {
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
        if torch.cuda.is_available()
        else {"num_workers": num_workers}
    )

    train_loader = DataLoader(
        GraphDINODataset(cfg, mode="train"),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate,
        **kwargs,
    )

    loaders = [train_loader]

    if cfg.data.eval_dataset is not None:
        val_dataset = GraphDINODataset(cfg, mode="eval")

        batch_size = (
            val_dataset.num_samples
            if len(val_dataset) < cfg.training.batch_size
            else cfg.training.batch_size
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=custom_collate,
            **kwargs,
        )
        loaders.append(val_loader)

    return tuple(loaders)

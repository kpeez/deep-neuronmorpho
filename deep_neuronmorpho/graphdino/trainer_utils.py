"""Utilities for training the GraphDINO model."""

from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, default_collate

from .data_utils import compute_laplacian_eigenvectors
from .dataset import GraphDINODataset


class GraphDINOLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for DINO-based graph representation learning.

    This module implements a training approach for graphs using DINO,
    mimicking the functionality of the original ssl_trainer.py Trainer class
    but leveraging PyTorch Lightning's framework.

    Attributes:
        model (nn.Module): The underlying graph neural network model.
        config (DictConfig): Configuration with model and training parameters.
        max_iter (int): Maximum number of training iterations.
        warmup_steps (int): Number of steps for learning rate warmup.
        lr_decay_steps (int): Steps after which learning rate decay is applied.
        init_lr (float): Initial learning rate.
        exp_decay (float): Exponential decay factor for learning rate.
        curr_iter (int): Current iteration counter.
    """

    def __init__(self, model: nn.Module, optimizer: Any, max_steps: int):
        """
        Initialize the GraphDINOLightningModule.

        Args:
            model (nn.Module): GraphDINO model.
            config (DictConfig): OmegaConf DictConfig object.
        """
        super().__init__()
        self.model = model
        self.max_iter = max_steps
        self.init_lr = optimizer
        self.optimizer = optimizer
        self.warmup_steps = self.max_iter // 50
        self.lr_decay_steps = self.max_iter // 5
        self.exp_decay = 0.5
        self.save_hyperparameters(ignore=["model"])
        # self.curr_iter = 0

    def configure_optimizers(self):
        """
        Set up the optimizer and a sequential learning rate scheduler
        that combines linear warmup with exponential decay.
        """
        optimizer = self.optimizer(params=self.model.parameters())
        optimizers = {"optimizer": optimizer}
        # warmup for the first `warmup_steps`
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-9, end_factor=1.0, total_iters=self.warmup_steps
        )
        # exponential decay after warmup
        gamma = self.exp_decay ** (1 / self.lr_decay_steps)
        decay_scheduler = ExponentialLR(optimizer, gamma=gamma)
        # warmup scheduler runs for `warmup_steps`, then the decay scheduler takes over.
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.warmup_steps],
        )
        optimizers["lr_scheduler"] = {
            "scheduler": lr_scheduler,
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return optimizers

    def set_lr(self):
        """Set the learning rate based on the current iteration."""
        optimizer = self.optimizers()

        if self.curr_iter < self.warmup_steps:
            lr = (self.curr_iter / self.warmup_steps) * self.init_lr
        else:
            lr = self.init_lr * self.exp_decay ** (
                (self.curr_iter - self.warmup_steps) / self.lr_decay_steps
            )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        self.log("learning_rate", lr, on_step=True, prog_bar=True)

        return lr

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Execute training step to process a batch of data.

        Args:
            batch (list): A batch of data containing feature and adjacency matrices.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The loss value.
        """
        f1, f2, a1, a2 = [x.float() for x in batch]
        l1 = compute_laplacian_eigenvectors(a1)
        l2 = compute_laplacian_eigenvectors(a2)
        loss = self.model(f1, f2, a1, a2, l1, l2)
        self.model.update_moving_average()
        self.log("train_loss", loss.mean(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Customize what gets saved in the checkpoint."""
        checkpoint["curr_iter"] = self.curr_iter

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Customize what gets loaded from the checkpoint."""
        if "curr_iter" in checkpoint:
            self.curr_iter = checkpoint["curr_iter"]

    def on_train_epoch_end(self):
        """Called at the end of a training epoch."""
        avg_loss = self.trainer.callback_metrics.get("train_loss_epoch", torch.tensor(0.0))
        print(f"Epoch {self.current_epoch} | Loss {avg_loss:.4f}")


class GraphDINODataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.kwargs = kwargs
        print(self.cfg, "cfg")
        print(f"using {self.cfg.num_workers} workers!!!!")

    def setup(self, stage: str):
        self.train_dataset = GraphDINODataset(self.cfg, mode="train")
        if self.cfg.eval_dataset is not None:
            self.val_dataset = GraphDINODataset(self.cfg, mode="eval")

    def train_dataloader(self):
        print("using batch size", self.cfg.batch_size)
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            collate_fn=custom_collate,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=custom_collate,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
        )


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

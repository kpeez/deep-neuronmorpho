"""Utilities for training the GraphDINO model."""

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
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
        automatic_optimization (bool): Always False as we handle optimization manually.
    """

    def __init__(self, model: nn.Module, config: DictConfig):
        """
        Initialize the GraphDINOLightningModule.

        Args:
            model (nn.Module): GraphDINO model.
            config (DictConfig): OmegaConf DictConfig object.
        """
        super().__init__()
        self.model = model
        self.cfg = config
        self.max_iter = self.cfg.training.max_steps
        self.init_lr = self.cfg.training.optimizer.lr
        self.exp_decay = 0.5
        self.warmup_steps = self.max_iter // 50
        self.lr_decay_steps = self.max_iter // 5

        self.curr_iter = 0

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.training.optimizer, params=self.model.parameters())
        optimizers = {"optimizer": optimizer}
        if self.cfg.training.scheduler is not None:
            scheduler = instantiate(self.cfg.training.scheduler, optimizer=optimizer)
            optimizers["lr_scheduler"] = scheduler

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
        optimizer = self.optimizers()
        f1, f2, a1, a2 = [x.float() for x in batch]

        l1 = compute_laplacian_eigenvectors(a1)
        l2 = compute_laplacian_eigenvectors(a2)

        self.set_lr()
        optimizer.zero_grad(set_to_none=True)
        loss = self.model(f1, f2, a1, a2, l1, l2)
        self.manual_backward(loss.sum())
        optimizer.step()
        self.model.update_moving_average()
        self.curr_iter += 1
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

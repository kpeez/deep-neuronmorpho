"""Trainer class for training a model."""

from typing import Any

import pytorch_lightning as pl
from torch import Tensor, nn
from torch_geometric.data import Batch


class ContrastiveGraphModule(pl.LightningModule):
    """
    A PyTorch Lightning module for contrastive learning on graph data.

    This module implements a contrastive learning approach for graph neural networks,
    using data augmentation and a contrastive loss function. It is designed to work
    with graph-structured data.

    Attributes:
        model (nn.Module): The underlying graph neural network model.
        loss_fn (nn.Module): The loss function.
        cfg_optimizer (Callable): Hydra configuration for optimizer.
        cfg_scheduler (Callable): Hydra configuration for scheduler.

    This module implements the following key functionalities:
        - Contrastive loss calculation using graph augmentation
        - Training and validation steps for contrastive learning
        - Configuration of optimizer and learning rate scheduler

        The class leverages PyTorch Lightning's LightningModule interface, implementing
        hooks for training, validation, and optimization that are automatically called
        by a PyTorch Lightning Trainer.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Any,
        scheduler: Any | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._best_train_loss = float("inf")
        self.logging = True

    @property
    def best_train_loss(self):
        return self._best_train_loss

    @best_train_loss.setter
    def best_train_loss(self, value):
        self._best_train_loss = value

    def training_step(self, batch: tuple[Batch, Batch]) -> Tensor:
        embed1, embed2 = self.model(batch[0]), self.model(batch[1])
        embed1 = self.all_gather(embed1, sync_grads=True)
        embed2 = self.all_gather(embed2, sync_grads=True)
        if embed1.dim() > 2:
            embed1 = embed1.view(-1, embed1.shape[-1])
            embed2 = embed2.view(-1, embed2.shape[-1])

        loss = self.loss_fn(embed1, embed2)
        graphs_per_view = batch[0].num_graphs

        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=self.logging,
            batch_size=graphs_per_view,
        )
        self.best_train_loss = min(self.best_train_loss, loss.item())
        self.log(
            "best_train_loss",
            self.best_train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=self.logging,
            batch_size=graphs_per_view,
        )
        return loss

    def on_training_epoch_end(self):
        avg_loss = self.trainer.callback_metrics["train_loss_epoch"]
        self.best_train_loss = min(self.best_train_loss, avg_loss.item())
        self.log(
            "train_loss_epoch",
            avg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=self.logging,
        )

    def configure_optimizers(self):
        """Instantiate optimizer and scheduler assuming configs are partial callables.

        Configs in `conf/training/optimizer/*.yaml` and `conf/training/scheduler/*.yaml`
        set `_partial_: true`, so Hydra passes callables here. We simply call them
        with the required runtime arguments.
        """
        optimizer = self.optimizer(params=self.model.parameters())
        optimizers: dict = {"optimizer": optimizer}

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            optimizers["lr_scheduler"] = scheduler

        return optimizers

"""Trainer class for training a model."""

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
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
        cfg (DictConfig): OmegaConf configuration for model and training parameters.
        loss_fn (nn.Module): The contrastive loss function.

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
        config: DictConfig,
        loss_fn: nn.Module,
    ):
        super().__init__()
        self.model = model
        self.cfg = config
        self.loss_fn = loss_fn
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

        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=self.logging,
        )
        self.best_train_loss = min(self.best_train_loss, loss.item())
        self.log(
            "best_train_loss",
            self.best_train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=self.logging,
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
        optimizer = instantiate(self.cfg.training.optimizer, params=self.model.parameters())
        optimizers = {"optimizer": optimizer}
        if self.cfg.training.scheduler is not None:
            scheduler = instantiate(self.cfg.training.scheduler, optimizer=optimizer)
            optimizers["lr_scheduler"] = scheduler

        return optimizers

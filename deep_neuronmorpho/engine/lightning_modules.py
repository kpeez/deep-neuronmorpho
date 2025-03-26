"""Trainer class for training a model."""

import pytorch_lightning as pl
import torch
from sklearn.svm import SVC
from torch import nn
from torch_geometric.data import Batch

from deep_neuronmorpho.data import GraphAugmenter
from deep_neuronmorpho.utils import Config

from .evaluation import repeated_kfold_eval
from .ntxent_loss import NTXEntLoss
from .trainer_utils import (
    create_optimizer,
    create_scheduler,
)


class ContrastiveGraphModule(pl.LightningModule):
    """
    A PyTorch Lightning module for contrastive learning on graph data.

    This module implements a contrastive learning approach for graph neural networks,
    using data augmentation and a contrastive loss function. It is designed to work
    with graph-structured data.

    Attributes:
        model (nn.Module): The underlying graph neural network model.
        cfg (Config): Configuration object containing model and training parameters.
        node_attrs (str): The key for node attributes in the graph data.
        loss_fn (NTXEntLoss): The contrastive loss function.
        augmenter (GraphAugmenter): Object for performing graph augmentations.
        _best_train_loss (float): The best (lowest) training loss observed.
        _best_val_acc (float): The best (highest) validation accuracy observed.
        validation_step_outputs (list): Stores outputs from validation steps.

    This module implements the following key functionalities:
        - Contrastive loss calculation using graph augmentation
        - Training and validation steps for contrastive learning
        - Accumulation of embeddings for k-fold cross-validation
        - Configuration of optimizer and learning rate scheduler

        The class leverages PyTorch Lightning's LightningModule interface, implementing
        hooks for training, validation, and optimization that are automatically called
        by a PyTorch Lightning Trainer.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        node_attrs: str = "nattrs",
        loss_fn: nn.Module | None = None,
    ):
        super().__init__()
        self.model = model
        self.cfg = config
        self.node_attrs = node_attrs
        self.loss_fn = (
            loss_fn if loss_fn is not None else NTXEntLoss(self.cfg.training.contra_loss_temp)
        )
        self.augmenter = GraphAugmenter(self.cfg.augmentation)
        self._best_train_loss = float("inf")
        self._best_val_acc = 0.0
        self.validation_step_outputs = []
        self.logging = True

    @property
    def best_train_loss(self):
        return self._best_train_loss

    @best_train_loss.setter
    def best_train_loss(self, value):
        self._best_train_loss = value

    @property
    def best_val_acc(self):
        return self._best_val_acc

    @best_val_acc.setter
    def best_val_acc(self, value):
        self._best_val_acc = value

    def compute_loss(self, batch_graphs: Batch) -> torch.Tensor:
        aug1_batch = self.augmenter.augment_batch(batch_graphs)
        aug1_embeds = self.model(aug1_batch, aug1_batch.ndata[self.node_attrs], is_training=True)
        aug2_batch = self.augmenter.augment_batch(batch_graphs)
        aug2_embeds = self.model(aug2_batch, aug2_batch.ndata[self.node_attrs], is_training=True)
        loss = self.loss_fn(aug1_embeds, aug2_embeds)
        return loss

    def training_step(self, batch: tuple[Batch, torch.Tensor]) -> torch.Tensor:
        batch_graphs = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss = self.compute_loss(batch_graphs)
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

    def validation_step(self, batch: tuple[Batch, torch.Tensor]) -> None:
        batch, labels = batch
        batch_feats = batch.ndata[self.node_attrs]
        model_output = self.model(batch, batch_feats, is_training=False)
        self.validation_step_outputs.append({"embeddings": model_output, "labels": labels})

    def on_validation_epoch_end(self):
        all_embeds = torch.cat([x["embeddings"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        eval_embeds = all_embeds.cpu().numpy()
        eval_labels = all_labels.cpu().numpy()
        clf = SVC()
        val_cv_acc, _ = repeated_kfold_eval(X=eval_embeds, y=eval_labels, model=clf, n_splits=5)
        self.log("val_cv_acc", val_cv_acc)
        self.best_val_acc = max(self.best_val_acc, val_cv_acc)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = create_optimizer(
            model=self.model,
            optimizer_name=self.cfg.training.optimizer,
            lr=self.cfg.training.lr,
        )
        if self.cfg.training.lr_scheduler is not None:
            scheduler = create_scheduler(
                kind=self.cfg.training.lr_scheduler.kind,
                optimizer=optimizer,
                step_size=self.cfg.training.lr_scheduler.step_size,
                factor=self.cfg.training.lr_scheduler.factor,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

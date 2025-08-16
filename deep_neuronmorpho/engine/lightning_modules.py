"""Trainer class for training a model."""

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.svm import SVC
from torch import Tensor, nn
from torch_geometric.data import Batch

from deep_neuronmorpho.data import augment_graph

from .evaluation import repeated_kfold_eval
from .ntxent_loss import NTXEntLoss


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
        config: DictConfig,
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

    def compute_loss(self, batch_graphs: Batch) -> Tensor:
        # TODO: use new dataset batching
        aug1_batch = augment_graph(batch_graphs)
        aug2_batch = augment_graph(batch_graphs)
        aug1_embeds = self.model(aug1_batch)
        aug2_embeds = self.model(aug2_batch)
        loss = self.loss_fn(aug1_embeds, aug2_embeds)
        return loss

    def training_step(self, batch: tuple[Batch, Tensor]) -> Tensor:
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

    def validation_step(self, batch: tuple[Batch, Tensor]) -> None:
        batch_graphs = batch[0] if isinstance(batch, (list, tuple)) else batch
        model_output = self.model(batch_graphs)
        labels = batch_graphs.y
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
        optimizer = instantiate(self.cfg.training.optimizer, params=self.model.parameters())
        optimizers = {"optimizer": optimizer}

        if self.cfg.training.scheduler is not None:
            scheduler = instantiate(self.cfg.training.scheduler, optimizer=optimizer)
            optimizers["lr_scheduler"] = scheduler

        return optimizers

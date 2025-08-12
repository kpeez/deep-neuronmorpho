"""Trainer class for training a model."""

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.svm import SVC
from torch import nn
from torch_geometric.data import Batch

from deep_neuronmorpho.data import augment_graph, compute_laplacian_eigenvectors

from .evaluation import repeated_kfold_eval
from .ntxent_loss import NTXEntLoss
from .trainer_utils import (
    create_optimizer,
    create_scheduler,
)


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

    def compute_loss(self, batch_graphs: Batch) -> torch.Tensor:
        aug1_batch = augment_graph(batch_graphs)
        aug1_embeds = self.model(aug1_batch, aug1_batch.ndata[self.node_attrs], is_training=True)
        aug2_batch = augment_graph(batch_graphs)
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
            optimizer_name=self.cfg.training.optimizer.name,
            lr=self.cfg.training.optimizer.lr,
        )
        if self.cfg.training.optimizer.scheduler is not None:
            scheduler = create_scheduler(
                kind=self.cfg.training.optimizer.scheduler["kind"],
                optimizer=optimizer,
                step_size=self.cfg.training.optimizer.scheduler["step_size"],
                factor=self.cfg.training.optimizer.scheduler["factor"],
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

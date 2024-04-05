"""Trainer class for training a model."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader
from sklearn.svm import SVC
from torch import nn

from deep_neuronmorpho.data import GraphAugmenter
from deep_neuronmorpho.utils import Config, ProgressBar, TrainLogger

from .evaluation import repeated_kfold_eval
from .ntxent_loss import NTXEntLoss
from .trainer_utils import (
    Checkpoint,
    get_optimizer,
    get_scheduler,
    setup_experiment_results,
)


class ContrastiveTrainer:
    """The ContrastiveTrainer class is a utility for training models with contrastive learning.

    Args:
        model (nn.Module): The model to train.
        config (ModelConfig): Configuration parameters for the model.
        dataloaders (Mapping[str, GraphDataLoader]): Dictionary of dataloaders for the
        contrastive training and classification evaluation datasets.
        device (torch.device): The device (CPU or GPU) to use for training.

    Methods:
        fit():
            Train the model for a given number of epochs.
        load_checkpoint(checkpoint_name: str):
            Load a checkpoint from file.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        dataloaders: Mapping[str, GraphDataLoader],
        device: torch.device | str,
        node_attrs: str = "nattrs",
    ):
        self.device = device
        self.cfg = config
        self.expt_name, expt_dir = setup_experiment_results(self.cfg)
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.node_attrs = node_attrs
        self.model_name = self.cfg.model.name
        self.loss_fn = NTXEntLoss(self.cfg.training.contra_loss_temp)
        self.augmenter = GraphAugmenter(self.cfg.augmentation)  # type: ignore
        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_name=self.cfg.training.optimizer,
            lr=self.cfg.training.lr,
        )
        if self.cfg.training.lr_scheduler is not None:
            self.lr_scheduler = get_scheduler(
                scheduler=self.cfg.training.lr_scheduler.kind,
                optimizer=self.optimizer,
                step_size=self.cfg.training.lr_scheduler.step_size,
                factor=self.cfg.training.lr_scheduler.factor,
            )
        self.logger = TrainLogger(f"{expt_dir}", expt_name=self.expt_name)
        self.checkpoint = Checkpoint(
            model=self.model,
            expt_name=self.expt_name,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            ckpt_dir=f"{expt_dir}/ckpts",
            device=self.device,
        )
        self.num_epochs = self.cfg.training.epochs
        self.best_train_loss = float("inf")
        self.eval_interval = self.cfg.training.eval_interval
        self.best_eval_acc = 0.0

    def _calculate_loss(self, batch_graphs: DGLGraph) -> float:
        """Calculate the loss for a batch.

        Args:
            batch_graphs (DGLGraph): Batch of graphs.

        Returns:
            float: The loss for the batch.
        """
        aug1_batch = self.augmenter.augment_batch(batch_graphs)
        aug1_embeds = self.model(aug1_batch, aug1_batch.ndata[self.node_attrs])
        aug2_batch = self.augmenter.augment_batch(batch_graphs)
        aug2_embeds = self.model(aug2_batch, aug2_batch.ndata[self.node_attrs])
        loss = self.loss_fn(aug1_embeds, aug2_embeds)

        return loss

    def train_step(self) -> float:
        """Train the model on the training set."""
        self.model.train()
        total_loss = 0.0
        for raw_batch in ProgressBar(self.dataloaders["contra_train"], desc="Processing batch:"):
            if isinstance(raw_batch, (list, tuple)):
                batch_graphs, _ = raw_batch
                batch_graphs = batch_graphs.to(self.device)
            else:
                batch_graphs = raw_batch.to(self.device)
            batch_feats = batch_graphs.ndata[self.node_attrs]
            batch_feats = batch_feats.to(self.device)
            loss = self._calculate_loss(batch_graphs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(self.dataloaders["contra_train"])

        return train_loss

    def eval_step(self) -> tuple[float, float, float, float]:
        """Evaluate the model on the validation set."""
        batch_embeds, batch_labels = [], []
        self.model.eval()
        with torch.inference_mode():
            for raw_batch in self.dataloaders["eval_train"]:
                batch, labels = raw_batch
                batch = batch.to(self.device)
                batch_feats = batch.ndata[self.node_attrs]
                batch_feats = batch_feats.to(self.device)
                model_output = self.model(batch, batch_feats)
                batch_embeds.append(model_output.detach().cpu().numpy())
                batch_labels.append(labels.detach().cpu().numpy())

        eval_embeds = np.vstack(batch_embeds)
        eval_labels = np.concatenate(batch_labels)

        clf = SVC()
        eval_metrics = repeated_kfold_eval(
            X=eval_embeds,
            y=eval_labels,
            model=clf,
        )

        return eval_metrics

    def fit(self, epochs: int | None = None, ckpt_file: str | Path | None = None) -> None:
        """Train the model for a given number of epochs.

        Iterate over the training and validation dataloaders for the given number of epochs,
        and save the model checkpoint after each evaluation step.

        Args:
            epochs (int, optional): Number of epochs to train for. By default, we train until max
            epochs (defined in model config). Defaults to None.
            ckpt_file (str, optional): Name of a model checkpoint file to load from `ckpts_dir`.
            If a checkpoint is provided, we resume training from that checkpoint. Defaults to None.
        """
        if ckpt_file is not None:
            self.checkpoint.load(ckpt_file)
            start_epoch = self.checkpoint.epoch if self.checkpoint.epoch is not None else 0
            self.logger.on_resume_training(ckpt_file, start_epoch)
        else:
            start_epoch = 0

        num_epochs = self.num_epochs - start_epoch if epochs is None else epochs
        self.logger.initialize(
            expt_name=self.expt_name,
            model_arch=self.cfg.model.model_dump(),
            num_epochs=num_epochs,
            device=self.device,
            random_state=self.cfg.training.random_state,
        )
        bad_epochs = 0
        for epoch in ProgressBar(range(start_epoch + 1, num_epochs + 1), desc="Training epochs:"):
            train_loss = self.train_step()
            self.logger.on_train_step(
                epoch=epoch,
                train_loss=train_loss,
                scheduler=self.lr_scheduler,
            )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.eval_interval is not None and epoch % self.eval_interval == 0:
                val_cv_acc, _, _, _ = self.eval_step()
                self.on_eval_step(val_cv_acc, epoch)
                if val_cv_acc > self.best_eval_acc:
                    self.best_eval_acc = val_cv_acc

            if epoch % self.cfg.training.save_every == 0:
                self.checkpoint.save(epoch=epoch, info_dict={"train_loss": train_loss})

            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if self.cfg.training.patience and bad_epochs > self.cfg.training.patience:
                self.logger.on_early_stop(epoch)
                break

        model_hparams = {
            "hidden_dim": self.cfg.model.hidden_dim,
            "output_dim": self.cfg.model.output_dim,
            "dropout": self.cfg.model.dropout_prob,
            **self.cfg.training.model_dump(),
        }
        self.logger.stop(
            params=model_hparams,
            metrics={
                "metric/best_train_loss": self.best_train_loss,
                "metric/best_eval_acc": self.best_eval_acc,
            },
        )

    def load_checkpoint(self, ckpt_name: str) -> None:
        """Load model checkpoint."""
        self.checkpoint.load(ckpt_name)

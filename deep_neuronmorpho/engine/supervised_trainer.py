"""Trainer for graph classification tasks."""

from pathlib import Path

import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler

from deep_neuronmorpho.engine.trainer_utils import (
    Checkpoint,
    get_optimizer,
    get_scheduler,
    setup_experiment_results,
)
from deep_neuronmorpho.utils import Config, ProgressBar, TrainLogger


class SupervisedTrainer:
    """A class for training GNN models for supervised learning.

    Args:
        model (nn.Module): The model to train.
        config (ModelConfig): Configuration parameters for the model.
        dataset (DGLDataset): The dataset to train on.
        device (torch.device): The device (CPU or GPU) to use for training.

    Methods:
        fit():
            Train the model for a given number of epochs.
        load_checkpoint(ckpt_name: str):
            Load a checkpoint from file.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        dataset: DGLDataset,
        device: torch.device | str,
        node_attrs: str = "nattrs",
    ):
        self.device = device
        self.cfg = config
        self.model = model.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.node_attrs = node_attrs
        self.dataset = dataset
        self.model_name = self.cfg.model.name
        self.batch_size = self.cfg.training.batch_size
        self.num_labels = len(self.dataset.glabel_dict)
        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_name=self.cfg.training.optimizer,
            lr=self.cfg.training.lr,
        )
        if (
            self.cfg.training.lr_scheduler is not None
            and self.cfg.training.lr_decay_steps is not None
            and self.cfg.training.lr_decay_rate is not None
        ):
            self.lr_scheduler = get_scheduler(
                scheduler=self.cfg.training.lr_scheduler.kind,
                optimizer=self.optimizer,
                step_size=self.cfg.training.lr_scheduler.step_size,
                factor=self.cfg.training.lr_scheduler.factor,
            )
        self.expt_name, expt_dir = setup_experiment_results(self.cfg)
        self.logger = TrainLogger(f"{expt_dir}/logs", expt_name=self.expt_name)
        self.checkpoint = Checkpoint(
            model=self.model,
            expt_name=self.expt_name,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            ckpt_dir=f"{expt_dir}/ckpts",
            device=self.device,
        )
        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=0,
            stratify=self.dataset.labels,
        )
        self.train_loader = GraphDataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            pin_memory=torch.cuda.is_available(),
        )
        self.val_loader = GraphDataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(val_idx),
            pin_memory=torch.cuda.is_available(),
        )
        self.num_train_samples = len(train_idx)
        self.num_val_samples = len(val_idx)
        self.num_epochs = self.cfg.training.epochs
        self.best_train_loss, self.best_val_loss = float("inf"), float("inf")
        self.best_train_acc, self.best_val_acc = 0.0, 0.0

    def train_step(self) -> tuple[float, float]:
        """Train the model on the training set."""
        self.model.train()
        train_loss, train_acc = 0.0, 0.0
        for batch_graphs, batch_labels in ProgressBar(self.train_loader, desc="Training batches:"):
            graphs = batch_graphs.to(self.device)
            labels = batch_labels.to(self.device, dtype=torch.long)
            logits = self.model(graphs, graphs.ndata[self.node_attrs])
            batch_loss = self.loss_fn(logits, labels)
            train_loss += batch_loss.item()
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            pred_labels = torch.argmax(logits, dim=1)
            train_acc += torch.sum(pred_labels == labels).item() / len(logits)

        train_loss /= len(self.train_loader)
        train_acc /= len(self.train_loader)

        return train_loss, train_acc

    def eval_step(self) -> tuple[float, float]:
        """Evaluate model performance on a given dataloader."""
        val_loss, val_acc = 0.0, 0.0
        self.model.eval()
        with torch.inference_mode():
            for batch_graphs, batch_labels in self.val_loader:
                graphs = batch_graphs.to(self.device)
                labels = batch_labels.to(self.device, dtype=torch.long)
                logits = self.model(graphs, graphs.ndata[self.node_attrs])
                batch_loss = self.loss_fn(logits, labels)
                val_loss += batch_loss.item()
                pred_labels = torch.argmax(logits, dim=1)
                val_acc += torch.sum(pred_labels == labels).item() / len(logits)

            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader)

        return val_loss, val_acc

    def fit(
        self,
        epochs: int | None = None,
        ckpt_file: str | Path | None = None,
    ) -> None:
        """Train the model for a given number of epochs.

        Iterate over the training and validation dataloaders for the given number of epochs,
        and save the model checkpoint if the validation loss improves.

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
            hparams=self.cfg.training.model_dump(),
            num_epochs=num_epochs,
            device=self.device,
            random_state=self.cfg.training.random_state,
        )
        bad_epochs = 0
        for epoch in ProgressBar(range(start_epoch + 1, num_epochs + 1), desc="Training epochs:"):
            train_loss, train_acc = self.train_step()
            self.logger.on_train_step(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                scheduler=self.lr_scheduler,
            )
            val_loss, val_acc = self.eval_step()
            self.logger.on_eval_step(
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            self.lr_scheduler.step()

            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
            if train_acc > self.best_train_acc:
                self.best_train_acc = train_acc
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                bad_epochs = 0
            else:
                bad_epochs += 1
            if self.cfg.training.patience and bad_epochs > self.cfg.training.patience:
                self.logger.on_early_stop(epoch)
                break
            info_dict = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
            self.checkpoint.save(epoch=epoch, info_dict=info_dict)

        self.logger.stop()

    def load_checkpoint(self, ckpt_name: str) -> None:
        """Load model checkpoint."""
        self.checkpoint.load(ckpt_name)

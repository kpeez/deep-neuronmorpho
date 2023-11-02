"""Supervised learning version of MACGNN."""
from pathlib import Path

import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from deep_neuronmorpho.engine.trainer_utils import (
    Checkpoint,
    get_optimizer,
    get_scheduler,
    setup_experiment_results,
)
from deep_neuronmorpho.utils import Config, EventLogger, ProgressBar


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
    ):
        self.device = device
        self.cfg = config
        self.model = model.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.model_name = self.cfg.model.name
        self.batch_size = self.cfg.training.batch_size
        self.num_labels = len(self.dataset.glabel_dict)
        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_name=self.cfg.training.optimizer,
            lr=self.cfg.training.lr_init,
        )
        self.lr_scheduler = get_scheduler(
            scheduler=self.cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            decay_steps=self.cfg.training.lr_decay_steps,
            decay_rate=self.cfg.training.lr_decay_rate,
        )
        self.expt_name, expt_dir = setup_experiment_results(self.cfg)
        self.logger = EventLogger(f"{expt_dir}/logs", expt_name=self.expt_name)
        self.checkpoint = Checkpoint(
            model=self.model,
            expt_name=self.expt_name,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            ckpt_dir=f"{expt_dir}/ckpts",
            device=self.device,
            logger=self.logger,
        )
        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
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
        self.max_epochs = self.cfg.training.max_epochs
        self.best_train_loss, self.best_val_loss = float("inf"), float("inf")
        self.best_train_acc, self.best_val_acc = 0.0, 0.0

    def train_step(self) -> tuple[float, float]:
        """Train the model on the training set."""
        self.model.train()
        total_train_loss, total_val_loss = 0.0, 0.0
        # train loss
        for train_batched_graph, train_batched_labels in ProgressBar(
            self.train_loader, desc="Training batches:"
        ):
            train_graph = train_batched_graph.to(self.device)
            train_labels = train_batched_labels.to(self.device, dtype=torch.long)
            train_logits = self.model(train_graph)
            train_loss = self.loss_fn(train_logits, train_labels)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            total_train_loss += train_loss.item()
        total_train_loss /= len(self.train_loader)
        # val loss
        for val_batched_graph, val__batch_labels in ProgressBar(
            self.val_loader, desc="Validation batches:"
        ):
            val_graph = val_batched_graph.to(self.device)
            val_labels = val__batch_labels.to(self.device, dtype=torch.long)
            val_logits = self.model(val_graph)
            val_loss = self.loss_fn(val_logits, val_labels)
            total_val_loss += val_loss.item()
        total_val_loss /= len(self.val_loader)

        return total_train_loss, total_val_loss

    def evaluate(self, dataloader: GraphDataLoader) -> float:
        """Evaluate model performance on a given dataloader.

        Args:
            dataloader (GraphDataLoader): Dataloader to evaluate on.

        Returns:
            float: Accuracy of the model on the given dataloader.
        """
        self.model.eval()
        correct = 0
        for batched_graphs, batched_label in dataloader:
            graphs = batched_graphs.to(self.device)
            labels = batched_label.to(self.device)
            logits = self.model(graphs)
            _, preds = torch.max(logits, dim=1)
            correct += torch.sum(preds == labels).item()

        total_samples = (
            self.num_train_samples if dataloader == self.train_loader else self.num_val_samples
        )

        return correct / total_samples

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
            self.logger.message(f"Resuming training from checkpoint: {ckpt_file}")
            start_epoch = self.checkpoint.epoch if self.checkpoint.epoch is not None else 0
        else:
            start_epoch = 0

        writer = SummaryWriter(log_dir=self.logger.log_dir)
        num_epochs = self.max_epochs if epochs is None else epochs
        self.logger.message(
            f"Training {self.model_name} on '{self.device}' for {num_epochs - start_epoch} epochs."
        )
        bad_epochs = 0
        for epoch in ProgressBar(range(start_epoch + 1, num_epochs + 1), desc="Training epochs:"):
            train_loss, val_loss = self.train_step()
            writer.add_scalar("loss/train_loss", train_loss, epoch, new_style=True)
            writer.add_scalar("loss/val_loss", val_loss, epoch, new_style=True)
            self.lr_scheduler.step()
            self.logger.message(
                f"Epoch {epoch}/{num_epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"  # noqa: E501
            )

            train_acc, val_acc = self.evaluate(self.train_loader), self.evaluate(self.val_loader)
            writer.add_scalar("acc/train_acc", train_acc, epoch, new_style=True)
            writer.add_scalar("acc/val_acc", val_acc, epoch, new_style=True)
            self.logger.message(
                f"Epoch {epoch}/{num_epochs}: Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}"
            )
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs > self.cfg.training.patience:
                self.logger.message(
                    f"Stopping training after {epoch} epochs: Validation accuracy at plateau"
                )
                break

            info_dict = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
            self.checkpoint.save(epoch=epoch, info_dict=info_dict)
        writer.close()

    def load_checkpoint(self, ckpt_name: str) -> None:
        """Load model checkpoint."""
        self.checkpoint.load(ckpt_name)

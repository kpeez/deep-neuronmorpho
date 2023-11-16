"""Trainer class for training a model."""
from itertools import zip_longest
from pathlib import Path

import torch
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from ..data import GraphAugmenter
from ..utils import Config, EventLogger, ProgressBar
from .evaluation import evaluate_embeddings
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
        dataloaders (dict[str, GraphDataLoader]): Dictionary of dataloaders for the
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
        dataloaders: dict[str, GraphDataLoader],
        device: torch.device | str,
        node_attrs: str = "nattrs",
    ):
        self.device = device
        self.cfg = config
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.node_attrs = node_attrs
        self.model_name = self.cfg.model.name
        self.loss_fn = NTXEntLoss(self.cfg.training.contra_loss_temp)
        self.augmenter = GraphAugmenter(self.cfg.augmentation)  # type: ignore
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
        self.max_epochs = self.cfg.training.max_epochs
        self.best_train_loss = float("inf")
        self.eval_interval = self.cfg.training.eval_interval
        self.best_eval_acc = 0.0

    def _calculate_loss(self, batch_graphs: DGLGraph, batch_feats: Tensor) -> float:
        """Calculate the loss for a batch.

        Args:
            batch_graphs (DGLGraph): Batch of graphs.
            batch_feats (Tensor): Batch of node features.

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
            batch_graphs = raw_batch.to(self.device)
            batch_feats = batch_graphs.ndata[self.node_attrs]
            batch_feats = batch_feats.to(self.device)
            loss = self._calculate_loss(batch_graphs, batch_feats)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(self.dataloaders["contra_train"])

        return train_loss

    def eval_step(self) -> float:
        """Evaluate the model on the validation set."""
        tensor_embeddings: dict[str, list[Tensor]] = {"train": [], "test": []}
        tensor_labels: dict[str, list[Tensor]] = {"train": [], "test": []}
        self.model.eval()
        with torch.inference_mode():
            for raw_train_batch, raw_test_batch in zip_longest(
                self.dataloaders["eval_train"], self.dataloaders["eval_test"]
            ):
                train_batch, train_labels = raw_train_batch
                train_batch = train_batch.to(self.device)
                train_batch_feats = train_batch.ndata[self.node_attrs]
                train_batch_feats = train_batch_feats.to(self.device)
                tensor_embeddings["train"].append(self.model(train_batch, train_batch_feats))
                tensor_labels["train"].append(train_labels)
                if raw_test_batch is not None:
                    test_batch, test_labels = raw_test_batch
                    test_batch = test_batch.to(self.device)
                    test_batch_feats = test_batch.ndata[self.node_attrs]
                    test_batch_feats = test_batch_feats.to(self.device)
                    tensor_embeddings["test"].append(self.model(test_batch, test_batch_feats))
                    tensor_labels["test"].append(test_labels)

        embeddings = {
            dataset: torch.cat(embed, dim=0).detach().cpu().numpy()
            for dataset, embed in tensor_embeddings.items()
        }
        labels = {
            dataset: torch.cat(label, dim=0).detach().cpu().numpy()
            for dataset, label in tensor_labels.items()
        }
        eval_acc = evaluate_embeddings(embeddings=embeddings, targets=labels)

        return eval_acc

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
            self.logger.message(f"Resuming training from checkpoint: {ckpt_file}")
            start_epoch = self.checkpoint.epoch if self.checkpoint.epoch is not None else 0
        else:
            start_epoch = 0

        writer = SummaryWriter(log_dir=self.logger.log_dir)
        num_epochs = self.max_epochs if epochs is None else epochs
        self.logger.message(
            f"Training {self.expt_name} on '{self.device}' "
            f"for {num_epochs - start_epoch} epochs "
            f"with random_seed {self.cfg.training.random_seed}."
        )
        bad_epochs = 0
        for epoch in ProgressBar(range(start_epoch + 1, num_epochs + 1), desc="Training epochs:"):
            train_loss = self.train_step()
            writer.add_scalar("loss/train", train_loss, epoch, new_style=True)
            self.logger.message(f"Epoch {epoch}/{num_epochs}: Train Loss: {train_loss:.4f}")
            self.lr_scheduler.step()

            if epoch % self.eval_interval == 0:
                eval_acc = self.eval_step()

                if eval_acc > self.best_eval_acc:
                    self.best_eval_acc = eval_acc

                writer.add_scalar("acc/eval_acc", eval_acc, epoch, new_style=True)
                self.logger.message(
                    f"Epoch {epoch}/{num_epochs}: Benchmark Test accuracy: {eval_acc:.4f}"
                )
                self.checkpoint.save(
                    epoch=epoch,
                    info_dict={
                        "train_loss": train_loss,
                        "eval_acc": eval_acc,
                        "best_eval_acc": self.best_eval_acc,
                    },
                )

            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs > self.cfg.training.patience:
                self.logger.message(
                    f"Stopping training after {epoch} epochs: Training loss at plateau"
                )
                break

        writer.close()

    def load_checkpoint(self, ckpt_name: str) -> None:
        """Load model checkpoint."""
        self.checkpoint.load(ckpt_name)

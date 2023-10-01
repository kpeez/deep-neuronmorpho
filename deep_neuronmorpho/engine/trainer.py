"""Trainer class for training a model."""
import shutil
from datetime import datetime
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
from .trainer_utils import Checkpoint, get_optimizer, get_scheduler


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
        load_checkpoint():
            Load the best model checkpoint from disk.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        dataloaders: dict[str, GraphDataLoader],
        device: torch.device | str,
    ):
        self.device = device
        self.cfg = config
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.model_name = self.cfg.model.name
        self.loss_fn = NTXEntLoss(self.cfg.training.contra_loss_temp)
        self.augmenter = GraphAugmenter(self.cfg.augmentation)
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

        timestamp = datetime.now().strftime("%Y_%m_%d_%Hh_%Mm")
        expt_name = f"{self.model_name}-{timestamp}"
        expt_dir = Path(self.cfg.dirs.expt_results) / expt_name
        for result in ["ckpts", "logs"]:
            result_dir = Path(expt_dir / result)
            if result_dir.exists() is False:
                result_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.cfg.config_file, expt_dir / f"{Path(self.cfg.config_file).stem}.yml")

        self.logger = EventLogger(expt_dir / "logs", expt_name=expt_name)
        self.checkpoint = Checkpoint(
            model=self.model,
            expt_name=expt_name,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            ckpt_dir=expt_dir / "ckpts",
            device=self.device,
            logger=self.logger,
        )

        self.max_epochs = self.cfg.training.max_epochs
        self.best_train_loss = float("inf")
        # self.eval_targets = get_eval_targets(self.cfg)
        self.eval_interval = self.cfg.training.eval_interval
        self.best_eval_acc = 0.0

    def _calculate_loss(self, batch: DGLGraph) -> float:
        """Calculate the loss for a batch.

        Args:
            batch (DGLGraph): Batch of graphs.

        Returns:
            float: The loss for the batch.
        """
        ypred = self.model(batch)
        augmented_batch = self.augmenter.augment_batch(batch)
        ypred_aug = self.model(augmented_batch)
        loss = self.loss_fn(ypred, ypred_aug)

        return loss

    def train_step(self) -> float:
        """Train the model on the training set."""
        self.model.train()
        total_loss = 0.0
        for raw_batch in ProgressBar(self.dataloaders["contra_train"], desc="Processing batch:"):
            batch = raw_batch.to(self.device)
            loss = self._calculate_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(self.dataloaders["contra_train"])

        return train_loss

    def eval_step(self) -> float:
        """Evaluate the model on the validation set."""
        embeddings: dict[str, list[Tensor]] = {"train": [], "test": []}
        labels: dict[str, list[Tensor]] = {"train": [], "test": []}
        self.model.eval()
        with torch.inference_mode():
            for raw_train_batch, raw_test_batch in zip_longest(
                self.dataloaders["eval_train"], self.dataloaders["eval_test"]
            ):
                train_batch, train_labels = raw_train_batch
                train_batch = train_batch.to(self.device)
                embeddings["train"].append(self.model(train_batch))
                labels["train"].append(train_labels)
                if raw_test_batch is not None:
                    test_batch, test_labels = raw_test_batch
                    test_batch = test_batch.to(self.device)
                    embeddings["test"].append(self.model(test_batch))
                    labels["test"].append(test_labels)

        embeddings = {
            dataset: torch.cat(embed, dim=0).detach().cpu().numpy()
            for dataset, embed in embeddings.items()
        }
        labels = {
            dataset: torch.cat(label, dim=0).detach().cpu().numpy()
            for dataset, label in labels.items()
        }
        eval_acc = evaluate_embeddings(embeddings=embeddings, targets=labels)

        return eval_acc

    def fit(self, epochs: int | None = None, ckpt_file: str | Path | None = None) -> None:
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
                    # model_name=self.model_name,
                    epoch=epoch,
                    train_loss=train_loss,
                    eval_acc=eval_acc,
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

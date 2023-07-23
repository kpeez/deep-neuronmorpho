"""Trainer class for training a model."""
from itertools import zip_longest
from pathlib import Path

import torch
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from ..data import GraphAugmenter
from ..utils import ModelConfig, ProgressBar, TrainLogger
from .evaluation import evaluate_embeddings, get_eval_targets
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
        config: ModelConfig,
        dataloaders: dict[str, GraphDataLoader],
        device: torch.device | str,
    ):
        self.cfg = config
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.model_name = self.cfg.model.name
        self.loss_fn = NTXEntLoss()
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
        self.device = device
        self.logger = TrainLogger(self.log_dir, session="train")
        self.checkpoint = Checkpoint(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.ckpt_dir,
            self.device,
            self.logger,
        )
        self.best_train_loss = float("inf")
        self.best_eval_acc = 0.0

    def _calculate_loss(self, batch: DGLGraph) -> float:
        """Calculate the loss for a batch.

        Args:
            batch (DGLGraph): Batch of graphs.

        Returns:
            float: The loss for the batch.
        """
        ypred = self.model(batch)
        augmented_batch = self.augmenter.augment_batch(batch).to(self.device)
        ypred_aug = self.model(augmented_batch)
        loss = self.loss_fn(ypred, ypred_aug)

        return loss

    def train_step(self) -> float:
        """Train the model on the training set."""
        self.model.train()
        total_loss = 0.0
        for raw_batch in ProgressBar(self.dataloaders["contra_train"], desc="Processing batch:"):
            batch = raw_batch.to(self.device)
            self.logger.message("Calculating loss...")
            loss = self._calculate_loss(batch)
            self.logger.message("Backpropagating...")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(self.dataloaders["contra_train"])

        return train_loss

    def eval_step(self) -> float:
        """Evaluate the model on the validation set."""
        tensor_embeddings: dict[str, list[Tensor]] = {"train": [], "test": []}
        self.model.eval()
        with torch.inference_mode():
            for raw_train_batch, raw_test_batch in zip_longest(
                self.dataloaders["eval_train"], self.dataloaders["eval_test"]
            ):
                train_batch = raw_train_batch.to(self.device)
                test_batch = raw_test_batch.to(self.device)
                tensor_embeddings["train"].append(self.model(train_batch))
                tensor_embeddings["test"].append(self.model(test_batch))
        embeddings = {
            dataset: torch.cat(embed, dim=0).detach().cpu().numpy()
            for dataset, embed in tensor_embeddings.items()
        }
        eval_acc = evaluate_embeddings(embeddings=embeddings, targets=self.eval_targets)

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
            self.logger.message(f"Resuming training from checkpoint: {ckpt_file}")
            self.checkpoint.load(ckpt_file, self.model_name)

        writer = SummaryWriter(self.log_dir)
        epochs = self.max_epochs if epochs is None else epochs
        self.logger.message(f"Training model on '{self.device}' for {epochs} epochs...")
        bad_epochs = 0
        for epoch in ProgressBar(range(1, epochs + 1), desc="Training epochs:"):
            train_loss = self.train_step()
            writer.add_scalar("loss/train", train_loss, epoch, new_style=True)
            self.logger.message(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.4f}")
            self.lr_scheduler.step()

            if epoch % self.eval_interval == 0:
                eval_acc = self.eval_step()

                if eval_acc > self.best_eval_acc:
                    self.best_eval_acc = eval_acc

                writer.add_scalar("acc/eval_acc", eval_acc, epoch, new_style=True)
                self.logger.message(
                    f"Epoch {epoch}/{epochs}: Benchmark Test accuracy: {eval_acc:.4f}"
                )
                self.checkpoint.save(
                    epoch=epoch,
                    train_loss=train_loss,
                    eval_acc=eval_acc,
                    model_name=self.model_name,
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

    @property
    def eval_interval(self) -> int:
        """Returns the number of epochs between model evaluations."""
        return self.cfg.training.eval_interval

    @property
    def max_epochs(self) -> int:
        """Returns the maximum number of epochs to train for."""
        return self.cfg.training.max_epochs

    @property
    def eval_targets(self) -> dict[str, NDArray]:
        """Get target labels for evaluation training and testing sets."""
        targets = get_eval_targets(self.cfg)

        return targets

    @property
    def ckpt_dir(self) -> Path:
        """Returns the directory where checkpoints are saved."""
        ckpt_dir = Path(self.cfg.output.ckpt_dir)
        if ckpt_dir.exists() is False:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        return ckpt_dir

    @property
    def log_dir(self) -> Path:
        """Returns the directory where logs are saved."""
        log_dir = Path(self.cfg.output.log_dir)
        if log_dir.exists() is False:
            log_dir.mkdir(parents=True, exist_ok=True)

        return log_dir

    def load_checkpoint(self, ckpt_name: str) -> None:
        """Load model checkpoint."""
        self.checkpoint.load(ckpt_name, self.model_name)

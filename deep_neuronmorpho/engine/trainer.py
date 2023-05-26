"""Trainer class for training a model."""
from itertools import zip_longest
from pathlib import Path
from typing import Any

import torch
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader
from numpy.typing import NDArray
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter

from ..data import GraphAugmenter
from ..utils import ModelConfig, ProgressBar, TrainLogger
from . import NTXEntLoss, evaluate_embeddings, get_eval_targets


class ContrastiveTrainer:
    """The ContrastiveTrainer class is a utility for training models with contrastive learning.

    Args:
        model (nn.Module): The model to train.
        config (ModelConfig): Configuration parameters for the model.
        dataloaders (dict[str, GraphDataLoader]): Dictionary of dataloaders for the
        contrastive training and classification evaluation datasets.
        device (torch.device): The device (CPU or GPU) to use for training.

    Methods:
        train_step() -> float:
            Train the model on the training set for one epoch and return the loss.
        eval_step() -> float:
            Evaluate the model on the validation set for one epoch and return the loss.
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
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.dataloaders = dataloaders
        self.loss_fn = NTXEntLoss()
        self.augmenter = GraphAugmenter(self.config.augmentation)
        self.optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_scheduler()
        self.device = device
        self.logger = TrainLogger(self.log_dir, session="train")
        self.best_train_loss = float("inf")
        self.best_eval_acc = 0.0

    def _get_optimizer(self) -> optim.Optimizer:
        optimizer_name = self.config.training.optimizer
        learning_rate = self.config.training.lr_init
        try:
            if optimizer_name.lower() == "adam":
                return torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            elif optimizer_name.lower() == "sgd":
                return torch.optim.SGD(self.model.parameters(), lr=learning_rate)
            elif optimizer_name.lower() == "rmsprop":
                return torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
            else:
                raise ValueError
        except ValueError:
            self.logger.message(
                f"Optimizer {optimizer_name} not supported. Defaulting to SGD.", level="warning"
            )
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _get_scheduler(self) -> Any:
        if self.config.training.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.lr_decay_steps,
                gamma=self.config.training.lr_decay_rate,
            )
        else:
            raise ValueError(f"Scheduler '{self.config.training.lr_scheduler}' not recognized")

    def _calculate_loss(self, batch: DGLGraph) -> float:
        """Calculate the loss for a batch.

        Args:
            batch (DGLGraph): Batch of graphs.

        Returns:
            float: The loss for the batch.
        """
        augmented_batch = self.augmenter.augment_batch(batch)
        ypred = self.model(batch)
        ypred_aug = self.model(augmented_batch)
        loss = self.loss_fn(ypred, ypred_aug)
        return loss

    def _process_batch(self, dataloader: GraphDataLoader, train_mode: bool = True) -> float:
        """Process a batch from dataloader.

        Args:
            dataloader (GraphDataLoader): DataLoader for the training/validation set.
            train_mode (bool, optional): Use train mode or eval mode. Defaults to True.

        Returns:
            float: The average loss over the batch.
        """
        total_loss = 0.0
        for raw_batch in ProgressBar(dataloader, desc="Processing batch:"):
            batch = raw_batch.to(self.device)
            loss = self._calculate_loss(batch)
            if train_mode:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def train_step(self) -> float:
        """Train the model on the training set."""
        self.model.train()
        train_loss = self._process_batch(self.dataloaders["contra_train"], train_mode=True)
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

    def fit(self, epochs: int | None = None) -> None:
        """Train the model for a given number of epochs.

        Iterate over the training and validation dataloaders for the given number of epochs,
        and save the model checkpoint if the validation loss improves.

        Args:
            epochs (int, optional): Number of epochs to train for. If 0, train until
                the validation loss stops improving or until the max number of epochs is reached.
                Defaults to 0 (max epochs in model config).

        """
        writer = SummaryWriter(self.log_dir)
        self.logger.message(f"Training model for {epochs} epochs...")
        epochs = self.max_epochs if epochs is None else epochs
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
                self.save_checkpoint(
                    epoch,
                    train_loss=train_loss,
                    eval_acc=eval_acc,
                )

            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs > self.config.training.patience:
                self.logger.message(
                    f"Stopping training after {epoch} epochs: Training loss at plateau"
                )
                break

        writer.close()

    @property
    def eval_interval(self) -> int:
        """Returns the number of epochs between model evaluations."""
        return self.config.training.eval_interval

    @property
    def max_epochs(self) -> int:
        """Returns the maximum number of epochs to train for."""
        return self.config.training.max_epochs

    @property
    def eval_targets(self) -> dict[str, NDArray]:
        """Get target labels for evaluation training and testing sets."""
        targets = get_eval_targets(self.config)

        targets_deleteme = {dataset: labels[:5] for dataset, labels in targets.items()}

        return targets_deleteme

    @property
    def checkpoint_dir(self) -> Path:
        """Returns the directory where checkpoints are saved."""
        chkpt_dir = Path(self.config.output.chkpt_dir)
        if chkpt_dir.exists() is False:
            chkpt_dir.mkdir(parents=True, exist_ok=True)

        return chkpt_dir

    @property
    def log_dir(self) -> Path:
        """Returns the directory where logs are saved."""
        log_dir = Path(self.config.output.log_dir)
        if log_dir.exists() is False:
            log_dir.mkdir(parents=True, exist_ok=True)

        return log_dir

    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        eval_acc: float,
    ) -> None:
        """Save model checkpoint.

        Args:
            epoch (int): Epoch number
            train_loss (float): Contrastive loss on the training set
            eval_acc (float): Classification accuracy on the evaluation test set.
        """
        chkpt_name = f"{self.config.model.name}_checkpoint_epoch_{epoch:03d}.pt"
        self.logger.message(f"Saving checkpoint: {self.checkpoint_dir}/{chkpt_name} ")
        chkpt_file = self.checkpoint_dir / chkpt_name
        torch.save(
            {
                self.config.model.name: self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch,
                "losses": {"contra_train": train_loss},
                "eval_acc": eval_acc,
            },
            chkpt_file,
        )

    def load_checkpoint(self, chkpt_name: str) -> None:
        """Load model checkpoint if it exists."""
        chkpt_file = self.checkpoint_dir / chkpt_name
        if chkpt_file.is_file():
            self.logger.message(f"Loading {chkpt_name} from: {self.checkpoint_dir}")
            checkpoint = torch.load(chkpt_file, map_location=self.device)
            self.model.load_state_dict(checkpoint[self.config.model.name])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.best_train_loss = checkpoint["losses"]["contra_train"]
            self.best_eval_acc = checkpoint["eval_acc"]
            self.logger.message(
                f"Loaded model at epoch={checkpoint['epoch']} with "
                f"validation accuracy: {checkpoint['eval_acc']:.4f}"
            )
        else:
            raise FileNotFoundError(f"Checkpoint file {chkpt_file} not found")

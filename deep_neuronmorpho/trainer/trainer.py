"""Trainer class for training a model."""
from pathlib import Path

import torch
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from ..data_loader import GraphAugmenter
from ..utils import ModelConfig, ProgressBar


class ContrastiveTrainer:
    """The ContrastiveTrainer class is a utility for training models with contrastive learning.

    Args:
        model (nn.Module): The model to train.
        config (ModelConfig): Configuration parameters for the model.
        train_dataloader (GraphDataLoader): DataLoader for the training dataset.
        val_dataloader (GraphDataLoader): DataLoader for the validation dataset.
        augmenter (GraphAugmenter): Augmenter for data augmentation.
        optimizer (optim.Optimizer): The optimizer to use for training.
        lr_scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        loss (nn.Module): The loss function to use for training.
        device (torch.device): The device (CPU or GPU) to use for training.

    Methods:
        train_step() -> float:
            Train the model on the training set for one epoch and return the loss.
        eval_step() -> float:
            Evaluate the model on the validation set for one epoch and return the loss.
        fit():
            Train the model for a given number of epochs.
        save_checkpoint(epoch: int, val_loss: float):
            Save a checkpoint for the current state of the model.
        load_checkpoint():
            Load the best model checkpoint from disk.

    Attributes:
        max_epochs() -> int:
            Returns the maximum number of epochs to train for.
        checkpoint_dir() -> Path:
            Returns the directory where checkpoints are saved.
        log_dir() -> Path:
            Returns the directory where logs are saved.

    Example:
        model = SomeModel()
        config = ModelConfig(config_file='config.yaml')
        train_dataloader = GraphDataLoader(...)
        val_dataloader = GraphDataLoader(...)
        augmenter = GraphAugmenter(...)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        loss = ContrastiveLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        trainer = ContrastiveTrainer(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            augmenter=augmenter,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss=loss,
            device=device
        )

        trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        train_dataloader: GraphDataLoader,
        val_dataloader: GraphDataLoader,
        augmenter: GraphAugmenter,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        loss: nn.Module,
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss = loss
        self.augmenter = augmenter
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.best_val_loss = float("inf")

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
        loss = self.loss(ypred, ypred_aug)
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
        train_loss = self._process_batch(self.train_dataloader, train_mode=True)
        return train_loss

    def eval_step(self) -> float:
        """Evaluate the model on the validation set."""
        self.model.eval()
        with torch.inference_mode():
            val_loss = self._process_batch(self.val_dataloader, train_mode=False)
        return val_loss

    def fit(self, num_epochs: int = 0) -> None:
        """Train the model for a given number of epochs.

        Iterate over the training and validation dataloaders for the given number of epochs,
        and save the model checkpoint if the validation loss improves.


        Args:
            num_epochs (int, optional): Number of epochs to train for. If 0, train until
                the validation loss stops improving or until the max number of epochs is reached.
                Defaults to 0.

        """
        writer = SummaryWriter(self.log_dir)

        bad_epochs = 0
        num_epochs = self.max_epochs if num_epochs == 0 else num_epochs
        for epoch in ProgressBar(range(num_epochs), desc="Training epochs:"):
            train_loss = self.train_step()
            val_loss = self.eval_step()
            writer.add_scalar("loss/train", train_loss, epoch, new_style=True)
            writer.add_scalar("loss/val", val_loss, epoch, new_style=True)
            print(
                f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            self.lr_scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs > self.config.training.patience:
                print(f"Stopping training after {epoch+1} epochs: Validation loss at plateau")
                break

        writer.close()

    @property
    def max_epochs(self) -> int:
        """Returns the maximum number of epochs to train for."""
        return self.config.training.max_epochs

    @property
    def checkpoint_dir(self) -> Path:
        """Returns the directory where checkpoints are saved."""
        return Path(self.config.config_file).parents[1] / self.config.logging.chkpt_dir

    @property
    def log_dir(self) -> Path:
        """Returns the directory where logs are saved."""
        return Path(self.config.config_file).parents[1] / self.config.logging.log_dir

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint.

        Args:
            epoch (int): Epoch number
            val_loss (float): Validation loss
        """
        chkpt_name = f"{self.config.model.name}_model-best.pt"
        chkpt_file = self.checkpoint_dir / chkpt_name
        torch.save(
            {
                self.config.model.name: self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            },
            chkpt_file,
        )

    def load_checkpoint(self) -> None:
        """Load model checkpoint if it exists."""
        chkpt_name = f"{self.config.model.name}_model-best.pt"
        chkpt_file = self.checkpoint_dir / chkpt_name

        if chkpt_file.is_file():
            print(f"Loading checkpoint from {chkpt_file}")
            checkpoint = torch.load(chkpt_file, map_location=self.device)

            self.model.load_state_dict(checkpoint[self.config.model.name])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.best_val_loss = checkpoint["val_loss"]

            print(
                f"Loaded model at epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}"
            )
        else:
            raise FileNotFoundError(f"Checkpoint file {chkpt_file} not found")

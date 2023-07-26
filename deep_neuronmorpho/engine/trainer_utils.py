"""Utilities for working with and tracking the training process."""

from pathlib import Path
from typing import Any

import torch
from dgl.dataloading import GraphDataLoader
from torch import nn, optim

from ..data import NeuronGraphDataset
from ..data.data_utils import create_dataloader
from ..utils import ModelConfig, TrainLogger


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    **kwargs: Any,
) -> optim.Optimizer:
    """Get an optimizer for the model.

    Args:
        model (nn.Module): The model to train.
        optimizer_name (str): The name of the optimizer to use.
        lr (float): The initial learning rate.
        **kwargs: Additional keyword arguments to pass to the optimizer.

    Returns:
        optim.Optimizer: The optimizer.
    """
    try:
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
        elif optimizer_name.lower() == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr, **kwargs)
        else:
            raise ValueError
    except ValueError:
        print(f"Optimizer {optimizer_name} not supported. Defaulting to Adam.")
        return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)


def get_scheduler(
    scheduler: str,
    optimizer: optim.Optimizer,
    decay_steps: int,
    decay_rate: float,
) -> Any:
    """Get learning rate scheduler for the optimizer.

    Args:
        scheduler (str): The name of the scheduler to use.
        optimizer (optim.Optimizer): The optimizer to use.
        decay_steps (int): The number of steps between each decay.
        decay_rate (float): The decay rate.

    Returns:
        optim.lr_scheduler: The learning rate scheduler.

    """
    if scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    else:
        raise ValueError(f"Scheduler '{scheduler}' not recognized")


def setup_dataloaders(
    conf: ModelConfig, datasets: list[str], **kwargs: Any
) -> dict[str, GraphDataLoader]:
    """Create dataloaders for contrastive training and evaluation datasets.

    Args:
        conf (ModelConfig): Model configuration.
        datasets (list[str]): List of dataset names from model configuration.
        kwargs: Additional keyword arguments to pass to the parent torch.utils.data.DataLoader
        arguments such as num_workers, pin_memory, etc.

    Returns:
        dict[str, GraphDataLoader]: Dictionary of dataloaders for each dataset.
    """
    data_dir = conf.datasets.root
    graph_datasets = {
        dataset: NeuronGraphDataset(data_dir, getattr(conf.datasets, dataset))
        for dataset in datasets
    }

    dataloaders = {
        dataset: create_dataloader(graph_dataset, batch_size=1024, shuffle=False, **kwargs)
        if "eval" in dataset
        else create_dataloader(graph_dataset, conf.training.batch_size, shuffle=True, **kwargs)
        for dataset, graph_dataset in graph_datasets.items()
    }

    return dataloaders


class Checkpoint:
    """Utility for saving and loading model checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: Any,
        ckpt_dir: str | Path,
        device: str | torch.device,
        logger: TrainLogger,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.logger = logger

    def save(self, epoch: int, train_loss: float, eval_acc: float, model_name: str) -> None:
        """Save model checkpoint.

        Args:
            epoch (int): Epoch number
            train_loss (float): Contrastive loss on the training set
            eval_acc (float): Classification accuracy on the evaluation test set.
            model_name (str): Name of the model
        """
        chkpt_name = f"{model_name}_checkpoint_epoch_{epoch:03d}.pt"
        self.logger.message(f"Saving checkpoint: {self.ckpt_dir}/{chkpt_name} ")
        chkpt_file = Path(self.ckpt_dir) / chkpt_name
        torch.save(
            {
                model_name: self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch,
                "losses": {"contra_train": train_loss},
                "eval_acc": eval_acc,
            },
            chkpt_file,
        )

    def load(self, chkpt_name: str | Path, model_name: str) -> None:
        """Load model checkpoint if it exists."""
        chkpt_file = Path(self.ckpt_dir) / chkpt_name
        if chkpt_file.is_file():
            self.logger.message(f"Loading {chkpt_name} from: {self.ckpt_dir}")
            checkpoint = torch.load(chkpt_file, map_location=self.device)
            self.model.load_state_dict(checkpoint[model_name])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.logger.message(
                f"Loaded model at epoch={checkpoint['epoch']} with "
                f"validation accuracy: {checkpoint['eval_acc']:.4f}"
            )
        else:
            raise FileNotFoundError(f"Checkpoint file {chkpt_file} not found")

"""Utilities for working with and tracking the training process."""

import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim

from deep_neuronmorpho.data import NeuronGraphDataset
from deep_neuronmorpho.models import MACGNN, MACGNNv2
from deep_neuronmorpho.utils import Config
from deep_neuronmorpho.utils.model_config import Model

from .ntxent_loss import NTXEntLoss


def setup_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed (int, optional): The random seed to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_optimizer(
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


def create_scheduler(
    optimizer: optim.Optimizer,
    kind: str,
    step_size: int,
    factor: float | int,
) -> Any:
    """Get learning rate scheduler for the optimizer.

    Args:
        optimizer (optim.Optimizer): The optimizer to use.
        kind (str): The name of the scheduler to use.
        step_size (int): The step size for the scheduler.
        factor (float | int): The factor for the scheduler. Typically < 1 for step and > 1 for cosine annealing.

    Returns:
        optim.lr_scheduler: The learning rate scheduler.

    """
    if kind == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=factor,
        )
    elif kind == "cosine":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=step_size,
            T_mult=int(factor),
            eta_min=1e-6,
        )
    else:
        raise ValueError(f"Scheduler '{kind}' not recognized")


def create_dataloader(
    graph_dataset: DGLDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs: Any,
) -> GraphDataLoader:
    """Create dataloaders for training and validation datasets.

    Args:
        graph_dataset (DGLDataset): Graph dataset.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the training data. Defaults to True.
        drop_last (bool): Whether to drop the last batch if it is smaller than the batch size.
        kwargs: Additional keyword arguments to pass to the parent torch.utils.data.DataLoader
        arguments such as num_workers, pin_memory, etc.

    Returns:
        GraphDataLoader: Dataloader of graph dataset.
    """
    graph_loader = GraphDataLoader(
        graph_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **kwargs,
    )

    return graph_loader


def setup_dataloaders(
    conf: Config, datasets: Sequence[str], **kwargs: Any
) -> dict[str, GraphDataLoader]:
    """Create dataloaders for contrastive training and evaluation datasets.

    Args:
        conf (Config): Model configuration.
        datasets (Sequence[str]): List of dataset names from model configuration.
        kwargs: Additional keyword arguments to pass to the parent torch.utils.data.DataLoader
        arguments such as num_workers, pin_memory, etc.

    Returns:
        dict[str, GraphDataLoader]: Dictionary of dataloaders for each dataset.
    """
    data_dir = conf.dirs.data
    graph_datasets = {
        dataset: NeuronGraphDataset(
            name=Path(f"{data_dir}/{getattr(conf.datasets, dataset)}"), from_file=True
        )
        for dataset in datasets
    }
    dataloaders = {}
    for dataset, graph_dataset in graph_datasets.items():
        dataloaders[dataset] = create_dataloader(
            graph_dataset=graph_dataset,
            batch_size=conf.training.batch_size if "eval" not in dataset else 16,
            shuffle="eval" not in dataset,
            drop_last="eval" not in dataset,
            persistent_workers="eval" in dataset,
            **kwargs,
        )

    return dataloaders


def setup_logging(conf: Config) -> tuple[TensorBoardLogger, Path]:
    runs = sorted(Path(conf.dirs.logging).glob(f"{conf.model.name}/run-*"))
    run_number = int(runs[-1].name.split("-")[1]) + 1
    expt_id = f"run-{run_number:03d}-{conf.model.name}-{conf.datasets.contra_train.split('-')[0]}"
    logger = TensorBoardLogger(save_dir=conf.dirs.logging, name=conf.model.name, version=expt_id)
    run_dir = Path(logger.log_dir)
    ckpts_dir = run_dir / "ckpts"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(conf.config_file, run_dir / f"{expt_id}_config.yaml")
    return logger, ckpts_dir


def log_hyperparameters(logger: TensorBoardLogger, conf: Config) -> None:
    """Log hyperparameters to TensorBoard.

    Args:
        logger (TensorBoardLogger): The TensorBoard logger.
        conf (Config): The configuration object.
    """
    hparams = {
        "model_name": conf.model.name,
        "optimizer": conf.training.optimizer,
        "learning_rate": conf.training.lr,
        "batch_size": conf.training.batch_size,
        "max_steps": conf.training.max_steps,
        "loss_function": conf.training.loss_fn,
        "loss_temp": conf.training.loss_temp,
    }
    if conf.training.lr_scheduler:
        lr_hparams = {
            "lr_scheduler": conf.training.lr_scheduler.kind,
            "lr_decay_steps": conf.training.lr_scheduler.step_size,
            "lr_decay_rate": conf.training.lr_scheduler.factor,
        }
        hparams.update(lr_hparams)
    logger.log_hyperparams(hparams)


def setup_callbacks(conf: Config, ckpts_dir: Path) -> list:
    model_checkpoint = ModelCheckpoint(
        dirpath=ckpts_dir,
        filename="{step:07d}-{train_loss:.2f}",
        every_n_train_steps=conf.training.logging_steps,
        save_last=True,
        save_top_k=-1,
    )
    early_stopping = (
        EarlyStopping(
            monitor="train_loss",
            patience=conf.training.patience,
            mode="min",
        )
        if conf.training.patience
        else None
    )
    callbacks = [model_checkpoint, early_stopping] if early_stopping else [model_checkpoint]
    return callbacks


def create_trainer(
    conf: Config,
    logger: TensorBoardLogger,
    callbacks: list[pl.Callback],
    **kwargs,
) -> pl.Trainer:
    return pl.Trainer(
        max_steps=conf.training.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        deterministic=True,
        logger=logger,
        num_sanity_val_steps=0,
        val_check_interval=conf.training.eval_interval,
        log_every_n_steps=conf.training.logging_steps,
        callbacks=callbacks,
        **kwargs,
    )


def create_model(name: str, conf: Model) -> torch.nn.Module:
    model_loaders = {
        "macgnnv2": MACGNNv2,
        "macgnn": MACGNN,
    }

    return model_loaders[name.lower()](conf)


def create_loss_fn(name: str, **kwargs):
    LOSSES = {
        "ntxent": NTXEntLoss,
    }
    return LOSSES[name.lower()](**kwargs)


class Checkpoint:
    """Utility for saving and loading model checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        expt_name: str,
        optimizer: optim.Optimizer,
        lr_scheduler: Any,
        ckpt_dir: str | Path,
        device: str | torch.device,
    ):
        self.model = model
        self.expt_name = expt_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.epoch = None
        self.info_dict: dict[str, Any] = {}

    def save(self, epoch: int, info_dict: dict[str, Any]) -> None:
        """Save model checkpoint.

        Args:
            epoch (int): Epoch number
            info_dict (dict): Dictionary of additional information to save in the checkpoint.
        """
        chkpt_name = f"{self.expt_name}-epoch_{epoch:04d}.pt"
        chkpt_file = Path(self.ckpt_dir) / chkpt_name
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "info_dict": {**info_dict},
        }

        torch.save(checkpoint, chkpt_file)

    def load(self, ckpt_file: str | Path) -> None:
        """Load model checkpoint if it exists.

        Args:
            ckpt_file (str): Name of the checkpoint file to load.
        """
        if Path(ckpt_file).is_file():
            checkpoint = torch.load(ckpt_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"]
            self.info_dict = checkpoint["info_dict"]
        else:
            raise FileNotFoundError(f"Checkpoint file {ckpt_file} not found")

    @staticmethod
    def load_model(
        ckpt_file: str | Path,
        model: torch.nn.Module,
        device: str | torch.device = "cpu",
    ) -> torch.nn.Module:
        """Load model state from checkpoint.

        Args:
            ckpt_file (str | Path): Checkpoint file to load.
            model (torch.nn.Module): Model to load the state into.
            device (str | torch.device, optional): Device to load the model state onto.

        Returns:
            torch.nn.Module: Model with updated state.
        """
        ckpt = torch.load(ckpt_file, map_location=torch.device(device))
        model.load_state_dict(ckpt["model"])

        return model

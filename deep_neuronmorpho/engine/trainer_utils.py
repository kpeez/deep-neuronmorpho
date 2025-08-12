"""Utilities for working with and tracking the training process."""

import shutil
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from deep_neuronmorpho.data import GraphDINODataset
from deep_neuronmorpho.models import MACGNN
from deep_neuronmorpho.utils import Config

from .ntxent_loss import NTXEntLoss


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
    graph_dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Create dataloaders for training and validation datasets.

    Args:
        graph_dataset (Dataset): Graph dataset.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the training data. Defaults to True.
        drop_last (bool): Whether to drop the last batch if it is smaller than the batch size.
        kwargs: Additional keyword arguments to pass to the parent torch.utils.data.DataLoader
        arguments such as num_workers, pin_memory, etc.

    Returns:
        GraphDataLoader: Dataloader of graph dataset.
    """
    graph_loader = DataLoader(
        graph_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **kwargs,
    )

    return graph_loader


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def build_dataloader(cfg: DictConfig):
    num_workers = cfg.training.num_workers if cfg.training.num_workers is not None else 0
    kwargs = (
        {
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
        if torch.cuda.is_available()
        else {"num_workers": num_workers}
    )

    train_loader = TorchDataLoader(
        GraphDINODataset(cfg, mode="train"),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate,
        **kwargs,
    )

    loaders = [train_loader]

    if cfg.data.eval_dataset is not None:
        val_dataset = GraphDINODataset(cfg, mode="eval")

        batch_size = (
            val_dataset.num_samples
            if len(val_dataset) < cfg.training.batch_size
            else cfg.training.batch_size
        )

        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=custom_collate,
            **kwargs,
        )
        loaders.append(val_loader)

    return tuple(loaders)


# TODO: refactor for GNNs
def setup_dataloaders(cfg: Config, datasets: Sequence[str], **kwargs: Any) -> dict[str, DataLoader]:
    """Create dataloaders for contrastive training and evaluation datasets.

    Args:
        cfg (Config): Model configuration.
        datasets (Sequence[str]): List of dataset names from model configuration.
        kwargs: Additional keyword arguments to pass to the parent torch.utils.data.DataLoader
        arguments such as num_workers, pin_memory, etc.

    Returns:
        dict[str, DataLoader]: Dictionary of dataloaders for each dataset.
    """
    graph_datasets = {
        dataset_name: GraphDINODataset(cfg=cfg, mode="train" if "train" in dataset_name else "eval")
        for dataset_name in datasets
    }
    dataloaders = {}
    for dataset, graph_dataset in graph_datasets.items():
        dataloaders[dataset] = create_dataloader(
            graph_dataset=graph_dataset,
            batch_size=cfg.training.batch_size if "eval" not in dataset else 16,
            shuffle="eval" not in dataset,
            drop_last="eval" not in dataset,
            persistent_workers=True,
            **kwargs,
        )

    return dataloaders


def setup_logging(cfg: Any) -> tuple[TensorBoardLogger, Path]:
    run_dir = Path(cfg.training.logging_dir) / cfg.model.name
    runs = sorted(run_dir.glob("run-*"))
    run_number = int(runs[-1].name.split("-")[1]) + 1 if runs else 1
    expt_id = f"run-{run_number:03d}"
    logger = TensorBoardLogger(
        save_dir=cfg.training.logging_dir,
        name=cfg.model.name,
        version=expt_id,
    )
    run_dir = Path(logger.log_dir)
    ckpts_dir = run_dir / "ckpts"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    # Copy or materialize the effective configuration for reproducibility
    with suppress(Exception):
        config_src = getattr(cfg, "config_file", None)
        if config_src is not None:
            shutil.copy(config_src, run_dir / f"{expt_id}_config.yaml")
        else:
            # If using Hydra DictConfig, save resolved config
            with suppress(Exception):
                OmegaConf.save(config=cfg, f=str(run_dir / f"{expt_id}_config.yaml"))

    return logger, ckpts_dir


def log_hyperparameters(logger: TensorBoardLogger, cfg: Config) -> None:
    """Log hyperparameters to TensorBoard.

    Args:
        logger (TensorBoardLogger): The TensorBoard logger.
        cfg (Config): The configuration object.
    """
    hparams = {
        "model_name": cfg.model.name,
        "optimizer": cfg.optimizer.name,
        "learning_rate": cfg.optimizer.lr,
        "batch_size": cfg.training.batch_size,
        "max_steps": cfg.training.max_steps,
    }
    if cfg.training.optimizer.scheduler:
        lr_hparams = {
            "lr_scheduler": cfg.training.optimizer.scheduler["kind"],
            "lr_decay_steps": cfg.training.optimizer.scheduler["step_size"],
            "lr_decay_rate": cfg.training.optimizer.scheduler["factor"],
        }
        hparams.update(lr_hparams)
    logger.log_hyperparams(hparams)


def setup_callbacks(cfg: Config, ckpts_dir: Path) -> list:
    model_checkpoint = ModelCheckpoint(
        dirpath=ckpts_dir,
        filename="{step:07d}-{train_loss:.2f}",
        every_n_train_steps=cfg.training.logging_steps,
        save_last=True,
        save_top_k=3,
    )

    callbacks = [model_checkpoint]

    return callbacks


def create_trainer(
    cfg: Config,
    logger: TensorBoardLogger,
    callbacks: list[pl.Callback],
    **kwargs,
) -> pl.Trainer:
    return pl.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        deterministic=True,
        logger=logger,
        num_sanity_val_steps=0,
        val_check_interval=cfg.training.eval_interval,
        log_every_n_steps=cfg.training.logging_steps,
        callbacks=callbacks,
        **kwargs,
    )


def create_model(name: str, cfg: Config) -> torch.nn.Module:
    model_loaders = {"macgnn": MACGNN}

    return model_loaders[name.lower()](cfg)


def create_loss_fn(name: str, **kwargs):
    LOSSES = {
        "ntxent": NTXEntLoss,
    }
    return LOSSES[name.lower()](**kwargs)

"""Utilities for working with and tracking the training process."""

import random
import shutil
from datetime import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch import nn, optim

from deep_neuronmorpho.data import NeuronGraphDataset
from deep_neuronmorpho.utils import Config


def setup_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed (int, optional): The random seed. Defaults to 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


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


def create_dataloader(
    graph_dataset: DGLDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
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
    conf: Config, datasets: list[str], **kwargs: Any
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
    data_dir = conf.dirs.data
    graph_datasets = {
        dataset: NeuronGraphDataset(
            graphs_path=data_dir,
            dataset_name=getattr(conf.datasets, dataset),
            dataset_path=data_dir,
        )
        for dataset in datasets
    }

    dataloaders = {
        dataset: create_dataloader(graph_dataset, batch_size=1024, shuffle=False, **kwargs)
        if "eval" in dataset
        else create_dataloader(graph_dataset, conf.training.batch_size, shuffle=True, **kwargs)
        for dataset, graph_dataset in graph_datasets.items()
    }

    return dataloaders


def setup_common_utilities(config_file: str, gpu: int | None = None) -> tuple[Config, str]:
    """Set up common utilities for model training.

    Args:
        config_file (str): Path to the configuration file.
        gpu (int | None, optional): Index of the GPU to use. Defaults to None.

    Returns:
        tuple[Config, str]: The configuration object and the device to use.
    """
    conf = Config.from_yaml(config_file=config_file)
    device = f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu"

    return conf, device


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


def generate_experiment_name() -> str:
    """Generate a random experiment name."""
    colors = [
        "red",
        "scarlet",
        "ruby",
        "coral",
        "pink",
        "orange",
        "peach",
        "apricot",
        "amber",
        "gold",
        "lemon",
        "honey",
        "green",
        "jade",
        "teal",
        "mint",
        "blue",
        "azure",
        "aqua",
        "indigo",
        "navy",
        "violet",
        "iris",
        "beige",
        "topaz",
        "silver",
        "slate",
        "gray",
        "onyx",
        "pearl",
    ]
    animals = [
        "albatross",
        "crow",
        "eagle",
        "falcon",
        "hawk",
        "heron",
        "owl",
        "pelican",
        "raven",
        "sparrow",
        "cheetah",
        "jaguar",
        "leopard",
        "lion",
        "panther",
        "tiger",
        "dolphin",
        "manta",
        "marlin",
        "orca",
        "seahorse",
        "shark",
        "trout",
        "bonobo",
        "gorilla",
        "cobra",
        "viper",
        "badger",
        "boar",
        "elephant",
        "hippo",
        "lynx",
        "pig",
        "wolf",
        "dragon",
    ]
    return f"{random.choice(colors)}_{random.choice(animals)}"


def setup_experiment_results(cfg: Config) -> tuple[str, str]:
    """Set up the experiment name and results directory.

    Args:
        cfg (Config): Configuration object with `dirs.expt_results` and `model.name` attributes.

    Returns:
        tuple[str, str]: The experiment name and results directory.
    """
    expt_id = generate_experiment_name()
    prev_expts = list(Path(cfg.dirs.results).glob(f"*{expt_id}*"))
    while prev_expts:
        expt_id = generate_experiment_name()
        prev_expts = list(Path(cfg.dirs.results).glob(f"*{expt_id}*"))

    expt_name = f"{cfg.model.name}-{expt_id}"
    timestamp = dt.now().strftime("%Y_%m_%d_%Hh_%Mm")
    expt_dir = Path(cfg.dirs.results) / f"{timestamp}-{expt_name}"
    result_dir = Path(expt_dir / "ckpts")
    if result_dir.exists() is False:
        result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.config_file, expt_dir / f"{expt_name}.yml")

    return expt_name, str(expt_dir)

"""Model configuration."""
from pathlib import Path

import yaml
from pydantic import BaseModel


class Dirs(BaseModel):
    """Paths to directories for storing data, metadata, and experiment results."""

    graph_data: str
    metadata: str
    expt_results: str


class Datasets(BaseModel):
    """Paths to datasets for training, validation, and testing."""

    contra_train: str
    eval_train: str
    eval_test: str


class Model(BaseModel):
    """Model architecture and hyperparameters."""

    name: str
    num_gnn_layers: int
    num_mlp_layers: int | None
    hidden_dim: int
    output_dim: int
    dropout_prob: float
    attrs_streams: dict[str, list[int]] | None
    use_edge_weight: bool | None
    learn_eps: bool | None
    neighbor_aggregation: str
    graph_pooling_type: str
    gnn_layer_aggregation: str
    stream_aggregation: str | None


class Training(BaseModel):
    """Parameters for training the model."""

    batch_size: int
    contra_loss_temp: float
    eval_interval: int
    max_epochs: int
    patience: int
    lr_init: float
    optimizer: str
    lr_scheduler: str
    lr_decay_steps: int
    lr_decay_rate: float


class AugmentationParams(BaseModel):
    """Parameters for data augmentation."""

    perturb: dict[str, float | int] | None = None
    rotate: dict[str, float | int] | None = {}
    drop_branches: dict[str, float | int] | None = None


class Augmentation(BaseModel):
    """Data augmentation methods and parameters."""

    order: list[str]
    params: AugmentationParams


class Config(BaseModel):
    """Main configuration class."""

    config_file: str | Path
    dirs: Dirs
    datasets: Datasets
    model: Model
    training: Training
    augmentation: Augmentation

    @classmethod
    def from_yaml(cls, config_file: str | Path) -> "Config":
        """Load a configuration from a YAML file."""
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict["config_file"] = config_file

        return cls(**config_dict)

    def __repr__(self) -> str:
        """String representation of a Config object."""
        items = []
        config_dict = self.model_dump()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value_repr = "\n".join(f"    {k}: {v}" for k, v in value.items())
                items.append(f"{key}:\n{value_repr}")
            else:
                items.append(f"{key}: {value!r}")

        config_items = "\n".join(items)

        return f"Config({config_items})"

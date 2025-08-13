"""Model configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    """Paths to datasets for training, validation, and testing."""

    train_dataset: str
    eval_dataset: str | None = None
    num_nodes: int | None = None
    feat_dim: int | None = None


class GNNConfig(BaseModel):
    """Model architecture and hyperparameters for GNN model."""

    name: str
    num_gnn_layers: int
    hidden_dim: int
    output_dim: int
    num_mlp_layers: int | None
    use_edge_weight: bool | None
    learn_eps: bool | None
    neighbor_aggregation: str
    graph_pooling_type: str
    gnn_layer_aggregation: str
    attrs_streams: dict[str, list[int]] | None
    stream_aggregation: str | None
    dropout_prob: float | None = None


class OptimizerConfig(BaseModel):
    """Optimizer configuration including learning rate and scheduler parameters."""

    name: str
    lr: float
    scheduler: dict[str, str | int | float] | None = None  # kind, step_size, factor


class Augmentations(BaseModel):
    """Data augmentation methods and parameters.

    Example:
    ```yaml
    augmentation:
      jitter: 0.1
      translate: 1.0
      rotate_axis: y
      num_drop_branches: 10
    ```
    The order of augmentations is determined by the order in the dictionary.
    """

    jitter: float | None = None
    translate: float | None = None
    rotation_axis: str | None = None
    num_drop_branches: int | None = None


class Training(BaseModel):
    """Parameters for training the model."""

    logging_dir: str
    max_steps: int | None = None
    batch_size: int
    loss_fn: str | None = None
    loss_temp: float | None = None
    eval_interval: int | None = None
    optimizer: OptimizerConfig
    num_workers: int | None = None
    random_state: int | None = None
    logging_steps: int = 100


class Config(BaseModel):
    """Main configuration class."""

    config_file: str | Path
    data: DataConfig
    model: GNNConfig
    training: Training
    augmentations: Augmentations | None = None

    @classmethod
    def load(cls, config_file: str | Path) -> "Config":
        """Load a configuration from a YAML file."""
        with open(config_file, "r", encoding="utf-8") as f:
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

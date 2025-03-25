"""Model configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    """Paths to datasets for training, validation, and testing."""

    dataset_root: str
    train: str
    evaluation: str | None = None


class GNNModel(BaseModel):
    """Model architecture and hyperparameters for GNN model."""

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


class GraphDINOConfig(BaseModel):
    """Model architecture and hyperparameters for GraphDINO model."""

    num_classes: int
    dim: int
    depth: int
    n_head: int
    pos_dim: int
    move_avg: float
    center_avg: float
    teacher_temp: float


class OptimizerConfig(BaseModel):
    """Optimizer configuration including learning rate and scheduler parameters."""

    name: str
    lr: float
    scheduler: dict[str, str | int | float] | None = None  # kind, step_size, factor


class Augmentation(BaseModel):
    """Data augmentation methods and parameters."""

    order: list[str]
    perturb: dict[str, float | int] | None = None
    rotate: dict[str, float | int] | None = {}
    drop_branches: dict[str, float | int] | None = None


class Training(BaseModel):
    """Parameters for training the model."""

    logging_dir: str
    max_steps: int | None = None
    batch_size: int
    loss_fn: str
    loss_temp: float | None = None
    eval_interval: int | None = None
    optimizer: OptimizerConfig
    patience: int | None = None
    random_state: int | None = None
    logging_steps: int = 100


class Config(BaseModel):
    """Main configuration class."""

    config_file: str | Path
    datasets: DatasetConfig
    model: GNNModel | GraphDINOConfig
    training: Training
    augmentation: Augmentation | None = None

    @classmethod
    def from_yaml(cls, config_file: str | Path) -> "Config":
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

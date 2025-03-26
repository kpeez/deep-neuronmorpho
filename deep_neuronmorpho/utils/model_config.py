"""Model configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    """Paths to datasets for training, validation, and testing."""

    dataset_root: str
    train: str
    evaluation: str | None = None


class GNNConfig(BaseModel):
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
    """Data augmentation methods and parameters.

    Example:
    ```yaml
    augmentation:
      perturb:
        jitter: 0.1
      rotate: {}
      drop_branches:
        prop: 0.2
        deg_power: 1.0
    ```
    The order of augmentations is determined by the order in the dictionary.
    """

    augmentations: dict[str, dict[str, float | int] | dict] = {}

    def get_augmentation_list(self) -> list[str]:
        """Return the list of augmentation names in the order they should be applied."""
        return list(self.augmentations.keys())

    def get_augmentation_params(self) -> dict:
        """Return all augmentation parameters."""
        return self.augmentations

    def model_post_init(self, __context):
        """Validate augmentation parameters after initialization."""
        super().model_post_init(__context)

        for aug_type, params in self.augmentations.items():
            if aug_type == "perturb":
                if not params or "jitter" not in params:
                    raise ValueError(f"Perturb augmentation requires 'jitter' parameter: {params}")
            elif aug_type == "rotate":
                # rotate doesn't need parameters, an empty dict is valid
                pass
            elif aug_type == "drop_branches":
                if not params or "prop" not in params:
                    raise ValueError(
                        f"Drop branches augmentation requires 'prop' parameter: {params}"
                    )
            else:
                raise ValueError(f"Unknown augmentation type: {aug_type}")


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
    model: GNNConfig | GraphDINOConfig
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

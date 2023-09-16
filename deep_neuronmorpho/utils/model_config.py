"""Parse model config file."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(config_file: str | Path) -> dict[str, Any]:
    """Load model config file.

    Args:
        config_file (str | Path): Path to config file.

    Returns:
        dict[str, Any]: Model configuration.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config


@dataclass(frozen=True)
class Config:
    """A configuration class that recursively creates a nested dataclass from a dictionary.

    Attributes:
        conf_dict (dict[str, Any | dict[str, Any]]): A dictionary of configuration parameters.
    """

    def __init__(self, conf_dict: dict[str, Any | dict[str, Any]]) -> None:
        for k, v in conf_dict.items():
            if isinstance(v, dict):
                object.__setattr__(self, k, Config(v))
            elif isinstance(v, str) and "path" in v:
                object.__setattr__(self, k, Path(v))
            else:
                object.__setattr__(self, k, v)

    def to_dict(self) -> dict:
        """Return the configuration dictionary representation of the object.

        Returns:
            dict[str, Any | dict[str, Any]]: A dict containing config parameters and their values.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """Return the string representation of a Config object."""
        items = []
        for key, value in self.__dict__.items():
            items.append(f"{key}: {value!r}")
        return "\n".join(items)


@dataclass(frozen=True)
class ModelConfig(Config):
    """A configuration class for the model.

    This class contains configuration parameters for the model, such as the root path,
    data, model architecture, training, data augmentation, and logging settings.

    To initialize a ModelConfig, provide the path to the config file.

    Example:
        cfg = ModelConfig('/path_to_config/myconfig.yml')
    """

    def __init__(self, config_file: str | Path) -> None:
        object.__setattr__(self, "config_file", str(config_file))
        config_dict = load_config(config_file)
        super().__init__(config_dict)

    def __post_init__(self) -> None:
        """Initialize nested `Config` objects for each configuration parameter.

        This method is automatically called after the object is created,
        and initializes nested `Config` objects for the following configuration parameters:
        - `data`
        - `model`
        - `training`
        - `augmentation`
        - `logging`

        Raises:
            TypeError: If any of the configuration parameters is not a dictionary.
        """

        def __post_init__(self: ModelConfig) -> None:
            """Initialize nested `Config` objects for each configuration parameter."""
            for key, value in self.__annotations__.items():
                if key != "root_path":
                    object.__setattr__(self, key, Config(value))

    def __repr__(self) -> str:
        """Return the string representation of a ModelConfig object."""
        items = []
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value_repr = "\n".join(f"    {k}: {v}" for k, v in value.__dict__.items())
                items.append(f"{key}:\n{value_repr}")
            else:
                items.append(f"{key}: {value!r}")
        return "\n".join(items)


def validate_model_config(config: dict[str, Any]) -> None:
    """Validate the model section of the config file.

    Args:
        config (dict[str, Any]): Model configuration.

    Raises:
        ValueError: Error if any of the required parameters are missing.
    """
    required_params = [
        "use_edge_weight",
        "learn_eps",
        "hidden_dim",
        "output_dim",
        "num_mlp_layers",
        "num_gnn_layers",
        "graph_pooling_type",
        "neighbor_aggregation",
        "gnn_layer_aggregation",
        "stream_aggregation",
        "dropout_prob",
        "attrs_streams",
    ]

    missing_params = [param for param in required_params if param not in config]

    if missing_params:
        raise ValueError(f"Missing required parameters in ModelConfig: {missing_params}")

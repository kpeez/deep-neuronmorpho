"""Utility functions for model-related tasks."""

from functools import partial

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


def aggregate_tensor(
    tensor_data: Tensor,
    aggregation_method: str = "cat",
    dim: int = -1,
    weights: Tensor | None = None,
) -> Tensor:
    """Aggregates tensor data along a given dimension according to the specified aggregation method.

    Args:
        tensor_data (Tensor): The tensor data to be aggregated.
        aggregation_method (str, optional): The method used to aggregate tensor data.
            Defaults to "cat".
        dim (int, optional): The dimension along which the aggregation is performed.
            Defaults to -1.
        weights (Optional[Tensor], optional): The weights for the weighted sum
            aggregation method. Required if the aggregation method is 'wsum'. Defaults to None.

    Returns:
        Tensor: The aggregated tensor data.

    Raises:
        NotImplementedError: If the aggregation method is not implemented.
    """

    def wsum(tensor_data: Tensor, weights: Tensor, dim: int = -1) -> Tensor:
        if weights is None:
            raise ValueError("weights cannot be None for weighted sum aggregation.")

        weighted_tensor = torch.mul(tensor_data, weights)
        return torch.sum(weighted_tensor, dim=dim)

    aggregation_method_dict = {
        "mean": partial(torch.mean, dim=dim),
        "max": partial(torch.max, dim=dim),
        "sum": partial(torch.sum, dim=dim),
        "wsum": partial(wsum, weights=weights, dim=dim),
    }

    if aggregation_method not in aggregation_method_dict and aggregation_method != "cat":
        raise NotImplementedError(f"Aggregation method '{aggregation_method}' is not implemented.")

    if aggregation_method == "cat":
        return torch.cat(tensor_data.unbind(dim=-1), dim=dim)
    else:
        aggregation_func = aggregation_method_dict.get(aggregation_method, lambda x: x)

        if aggregation_method == "max":
            return aggregation_func(tensor_data)[0]
        else:
            return aggregation_func(tensor_data)


class GlobalPooling(Module):
    """Class for global graph pooling of node features to create graph-level representation."""

    def __init__(self, pooling_type: str):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """Pool node features to create graph-level representation.

        Args:
            x (Tensor): Node features.
            batch (Tensor): Batch information for the nodes.

        Returns:
            Tensor: Graph-level representation.
        """
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "sum":
            return global_add_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        else:
            raise NotImplementedError(
                f"Graph pooling type '{self.pooling_type}' is not implemented."
            )


def compute_embedding_dim(
    hidden_dim: int,
    num_gnn_layers: int,
    num_streams: int,
    gnn_layer_aggregation: str,
    stream_aggregation: str,
) -> int:
    """Computes the embedding dimension based on the given parameters.

    Args:
        hidden_dim (int): The dimensionality of the hidden layers in the GNN.
        num_gnn_layers (int): The number of GNN layers in the model.
        num_streams (int): The number of input streams.
        gnn_layer_aggregation (str): The aggregation method used for the GNN layers.
        Possible values are ["sum", "mean", "max", "wsum", "cat"].
        stream_aggregation (str): The aggregation method used for the input streams.
        Possible values are ["sum", "mean", "max", "wsum", "cat"].

    Returns:
        int: The embedding dimension computed based on the given parameters.
    """
    if gnn_layer_aggregation == "cat" and stream_aggregation == "cat":
        embedding_dim = hidden_dim * num_gnn_layers * num_streams
    elif gnn_layer_aggregation == "cat":
        embedding_dim = hidden_dim * num_gnn_layers
    elif stream_aggregation == "cat":
        embedding_dim = hidden_dim * num_streams
    else:
        embedding_dim = hidden_dim
    return embedding_dim


def load_attrs_streams(attrs_streams: dict[str, list[int]] | None) -> dict[str, list[int]]:
    """Load attribute streams from a dictionary containing the range of indices.

    Args:
        attrs_streams (dict[str, str]): A dictionary where the keys are stream names,
        and the values are a tuple the index range (inclusive) in string format.

    Returns:
        dict[str, list[int]]: A dictionary where the keys are stream names,
        and the values are lists of indices corresponding to the given index range.
    """
    attrs_idxs = {}

    if attrs_streams is None:
        raise ValueError("attrs_streams cannot be None.")

    for name, stream in attrs_streams.items():
        start, end = stream
        attrs_idxs[name] = list(range(start, end + 1))

    return attrs_idxs

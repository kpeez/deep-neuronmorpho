"""Utility functions for model-related tasks."""

from functools import partial

import torch
from dgl.nn.pytorch import AvgPooling, MaxPooling, SumPooling
from torch import Tensor, nn


def create_pooling_layer(pooling_type: str) -> nn.Module:
    """Create a pooling layer for the given pooling type.

    Args:
        pooling_type (str): The pooling type ('sum', 'mean', or 'max').

    Returns:
        nn.Module: The created pooling layer.

    Raises:
        NotImplementedError: If the pooling type is not implemented.
    """
    pooling_layer_dict = {
        "sum": SumPooling,
        "mean": AvgPooling,
        "max": MaxPooling,
    }
    try:
        return pooling_layer_dict[pooling_type]()
    except KeyError as err:
        raise NotImplementedError(
            f"Graph pooling type '{pooling_type}' is not implemented."
        ) from err


def aggregate_tensors(
    tensor_data: Tensor,
    aggregation_method: str = "cat",
    dim: int = -1,
    weights: Tensor | None = None,
) -> torch.Tensor:
    """Aggregates tensor data according to the specified aggregation method.

    Args:
        tensor_data (torch.Tensor): The tensor data to be aggregated.
        aggregation_method (str, optional): The method used to aggregate tensor data.
            Defaults to "cat".
        dim (int, optional): The dimension along which the aggregation is performed.
            Defaults to -1.
        weights (Optional[torch.Tensor], optional): The weights for the weighted sum
            aggregation method. Required if the aggregation method is 'wsum'. Defaults to None.

    Returns:
        torch.Tensor: The aggregated tensor data.

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
        return aggregation_func(tensor_data)

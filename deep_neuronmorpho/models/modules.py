"""Modules used to create various models."""

from typing import Any

import torch
from dgl.nn.pytorch.conv import GINConv
from torch import nn


def linear_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """Linear block."""
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(),
    )


class MLP(nn.Module):
    """MLP with linear output layer.

    Args:
        input_dim (int): Input dimension.
        hidden_dim (int): Hidden layer dimension.
        output_dim (int): Output dimension.
        num_layers (int): Number of layers. There are num_layers - 2 hidden layers.

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        if num_layers == 1:  # no hidden layers
            self.layers.append(linear_block(input_dim, output_dim))
        else:  # at least one hidden layer
            self.layers.append(linear_block(input_dim, hidden_dim))
            for _layer in range(num_layers - 2):
                self.layers.append(linear_block(hidden_dim, hidden_dim))

            self.layers.append(linear_block(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass inputs through MLP layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size),
            where output_size is the size of the last layer in the MLP.
        """
        for layer in self.layers:
            x = layer(x)
        return x


def gin_block(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_mlp_layers: int,
    aggregator_type: str,
    **kwargs: Any,
) -> nn.Sequential:
    """GNN block."""
    node_mlp = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_mlp_layers,
    )
    return nn.Sequential(
        GINConv(node_mlp, aggregator_type, **kwargs),
        nn.BatchNorm1d(output_dim),
    )

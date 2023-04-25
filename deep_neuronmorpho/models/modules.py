"""Modules used to create various models."""


import torch
from torch import nn


def linear_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """Linear block."""
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(),
    )


class MLP(nn.Module):
    """Simple MLP implementation.

    Args:
        input_dim (int): Input dimension.
        hidden_dim (int): Hidden layer dimension.
        output_dim (int): Output dimension.
        num_layers (int): Number of layers. There are num_layers - 2 hidden layers.

    Attributes:
        mlp (nn.Sequential): The sequential container of the MLP layers.

    Methods:
        forward: Pass inputs through MLP layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        layers = []
        if num_layers == 1:  # no hidden layers
            layers.append(linear_block(input_dim, output_dim))
        else:  # at least one hidden layer
            layers.append(linear_block(input_dim, hidden_dim))
            for _layer in range(num_layers - 2):
                layers.append(linear_block(hidden_dim, hidden_dim))

            layers.append(linear_block(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass inputs through MLP layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size),
            where output_size is the size of the last layer in the MLP.
        """
        return self.mlp(x)

"""Modules used to create various models."""


import dgl
import torch
from dgl.nn.pytorch import GINConv
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


class GINBlock(nn.Module):
    """GINBlock represents a single GIN layer in the GNN.

    It applies the GINConv operation, followed by batch normalization, activation, and dropout.

    Args:
        input_dim (int): The input feature dimension.
        hidden_dim (int): The hidden feature dimension.
        num_mlp_layers (int): The number of layers in the MLP used in GINConv.
        aggregator_type (str): The aggregator type used in GINConv (e.g., 'sum', 'mean', or 'max').
        dropout_prob (float): The probability of dropout.
        learn_eps (bool, optional): Whether to learn the epsilon parameter in GINConv.
        Defaults to False.

    Attributes:
        node_mlp (MLP): The MLP used in GINConv.
        gin_conv (GINConv): The GINConv layer.
        bn (nn.BatchNorm1d): The batch normalization layer.
        activation (nn.ReLU): The activation function (ReLU).
        dropout (nn.Dropout): The dropout layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_mlp_layers: int,
        aggregator_type: str,
        dropout_prob: float,
        learn_eps: bool = False,
    ):
        super().__init__()

        self.node_mlp = MLP(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
        )
        self.gin_conv = GINConv(self.node_mlp, aggregator_type, init_eps=0, learn_eps=learn_eps)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        graph: dgl.DGLGraph,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the GINBlock.

        Args:
            x (torch.Tensor): The input node features.
            graph (dgl.DGLGraph): The input graph.
            edge_weight (torch.Tensor, optional): The edge weights. Defaults to None.

        Returns:
            torch.Tensor: The output node features after applying the GINBlock.
        """
        x = self.gin_conv(graph, x, edge_weight)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

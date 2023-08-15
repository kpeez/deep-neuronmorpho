"""Modules used to create various models."""


import dgl
from dgl.nn.pytorch import GINConv
from torch import Tensor, nn


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
        num_layers (int): Number of layers. There are num_layers - 1 hidden layers.

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
        for _layer in range(num_layers - 1):
            layers.append(linear_block(input_dim, hidden_dim))
            input_dim = hidden_dim
        layers.append(linear_block(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the MLP."""
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
        x: Tensor,
        graph: dgl.DGLGraph,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the GINBlock.

        Args:
            x (Tensor): The input node features.
            graph (dgl.DGLGraph): The input graph.
            edge_weight (Tensor, optional): The edge weights. Defaults to None.

        Returns:
            Tensor: The output node features after applying the GINBlock.
        """
        x = self.gin_conv(graph, x, edge_weight)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


def create_gin_layers(
    num_gnn_layers: int,
    input_dim: int,
    hidden_dim: int,
    num_mlp_layers: int,
    aggregation_type: str,
    dropout_prob: float,
    learn_eps: bool,
) -> nn.ModuleList:
    """Create a list of GNN layers (GINBlocks) for a given configuration.

    Args:
        num_gnn_layers (int): The number of GNN layers to create.
        input_dim (int): The input dimension of the first GINBlock layer.
        hidden_dim (int): The hidden dimension for all GINBlock layers.
        num_mlp_layers (int): The number of MLP layers within each GINBlock.
        aggregation_type (str): The type of aggregation for each GINBlock.
        dropout_prob (float): The dropout probability for each GINBlock.
        learn_eps (bool): Whether to learn the epsilon parameter in each GINBlock.

    Returns:
        nn.ModuleList: A list of GNN layers (GINBlocks) with the given configuration.
    """
    gnn_layers = nn.ModuleList()
    for layer in range(num_gnn_layers):
        input_dim = input_dim if layer == 0 else hidden_dim
        gnn_layers.append(
            GINBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_mlp_layers=num_mlp_layers,
                aggregator_type=aggregation_type,
                dropout_prob=dropout_prob,
                learn_eps=learn_eps,
            )
        )
    return gnn_layers

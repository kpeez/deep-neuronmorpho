"""Simple feed-forward MLP implementation."""
from torch import Tensor, nn


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
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ]
            )
            input_dim = hidden_dim
        layers.extend([nn.Linear(hidden_dim, output_dim)])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the MLP."""
        return self.mlp(x)

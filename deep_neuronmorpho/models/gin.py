"""Graph Isomorphism Network (GIN) model. from [Xu et al. 2019](https://arxiv.org/abs/1810.00826).

See example DGL implementation [here](https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py).

"""

import torch
from dgl import DGLGraph
from dgl.nn.pytorch import GINConv
from torch import Tensor, nn

from .mlp import MLP
from .model_utils import aggregate_tensor, create_pooling_layer


class GIN(nn.Module):
    r"""Graph isomorphism network.

    From [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826).

    At each layer, the node representations are updated by:
    - aggregating the representations of their neighbors
    - applying a multilayer perceptron (MLP) to the aggregated representation

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden feature dimension.
        output_dim (int): Output feature dimension.
        num_layers (int): Number of GIN layers.
        num_mlp_layers (int): Number of MLP layers in each GIN layer.
        neighbor_aggregation (str): Aggregation type for GINConv ['sum', 'mean', 'max'].
        graph_pooling (str): Graph pooling method ['sum', 'mean', 'max'] applied to each GIN layer.
        layer_aggregation (str): Aggregation method for the graph-level representations from each
        layer ['cat', 'sum', 'mean', 'max'].
        dropout_prob (float): Dropout probability applied to each GIN layer.
        learn_eps (bool): Use the learnable epsilon parameter in GINConv.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_mlp_layers: int,
        neighbor_aggregation: str = "sum",
        graph_pooling: str = "max",
        layer_aggregation: str = "cat",
        dropout_prob: float = 0.5,
        learn_eps: bool = False,
    ):
        super().__init__()
        self.layer_aggregation = layer_aggregation
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # linear functions for projecting all layers output_dim
        self.linear_prediction = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.PReLU()
        self.graph_pool = create_pooling_layer(graph_pooling)

        for layer in range(num_layers - 1):
            in_dim = input_dim if layer == 0 else hidden_dim
            mlp = MLP(in_dim, hidden_dim, hidden_dim, num_layers=num_mlp_layers)
            self.gin_layers.append(
                GINConv(
                    apply_func=mlp,
                    aggregator_type=neighbor_aggregation,
                    learn_eps=learn_eps,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.linear_prediction.append(nn.Linear(in_dim, output_dim))

    def forward(self, g: DGLGraph, h: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        """Forward pass of the GIN model.

        After applying the GIN layers, graph-level representations from each layer are obtained
        by applying a graph pooling layer. The graph-level representations from each layer are then
        aggregated using the specified aggregation method to obtain the final graph representation.

        Args:
            g (DGLGraph): Input graph from dataloader.
            h (Tensor): Tensor of node features with shape (num_nodes, input_dim).
            edge_weight (Tensor | None, optional): Tensor of edge weights. Defaults to None.

        Returns:
            Tensor: Graph-level representation of the input graph.
        """
        hidden_rep = [h]
        for layer, gin_conv in enumerate(self.gin_layers):
            h = gin_conv(g, h, edge_weight)
            h = self.batch_norms[layer](h)
            h = self.activation(h)
            hidden_rep.append(h)

        graph_reps = []
        for layer, h_rep in enumerate(hidden_rep):
            h_pooled = self.graph_pool(g, h_rep)
            graph_reps.append(self.dropout(self.linear_prediction[layer](h_pooled)))

        final_graph_rep = aggregate_tensor(
            torch.stack(graph_reps, dim=-1), aggregation_method=self.layer_aggregation
        )

        return final_graph_rep

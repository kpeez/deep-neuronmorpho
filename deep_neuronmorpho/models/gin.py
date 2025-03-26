"""Graph Isomorphism Network (GIN) model. from [Xu et al. 2019](https://arxiv.org/abs/1810.00826)."""

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn.conv.gin_conv import GINConv, GINEConv

from .mlp import MLP
from .model_utils import GlobalPooling, aggregate_tensor


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
        use_edge_weight (bool): Whether to use edge weights.
        edge_dim (int): Edge feature dimension if use_edge_weight is True.
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
        use_edge_weight: bool = False,
        edge_dim: int | None = None,
        graph_pooling: str = "max",
        layer_aggregation: str = "cat",
        dropout_prob: float = 0.0,
        learn_eps: bool = False,
    ):
        super().__init__()
        self.use_edge_weight = use_edge_weight
        self.edge_dim = edge_dim
        self.layer_aggregation = layer_aggregation
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # linear functions for projecting all layers output_dim
        self.linear_prediction = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.PReLU()
        self.graph_pool = GlobalPooling(graph_pooling)

        for layer in range(num_layers - 1):
            in_dim = input_dim if layer == 0 else hidden_dim
            mlp = MLP(in_dim, hidden_dim, hidden_dim, num_layers=num_mlp_layers)
            if self.use_edge_weight:
                self.gin_layers.append(GINEConv(mlp, train_eps=learn_eps, edge_dim=self.edge_dim))
            else:
                self.gin_layers.append(GINConv(mlp, train_eps=learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.linear_prediction.append(nn.Linear(in_dim, output_dim))

    def forward(self, g_batch: Batch, feat_index: list[int] | None = None) -> Tensor:
        """Forward pass of the GIN model.

        After applying the GIN layers, graph-level representations from each layer are obtained
        by applying a graph pooling layer. The graph-level representations from each layer are then
        aggregated using the specified aggregation method to obtain the final graph representation.

        Args:
            g_batch (Batch): PyG Batch object containing the input graphs.
            feat_index (list[int]): List of node feature indices to use for the GIN layers.

        Returns:
            Tensor: Graph-level representation with output shape is (batch_size, output_dim).
            Note that if layer_aggregation is 'cat', the output shape is
            (batch_size, num_layers * output_dim).
        """
        edge_index, edge_weight, batch = (g_batch.edge_index, g_batch.edge_attr, g_batch.batch)
        h = g_batch.x[:, feat_index] if feat_index is not None else g_batch.x
        hidden_rep = [h]
        for layer, gin_conv in enumerate(self.gin_layers):
            if self.use_edge_weight:
                h = gin_conv(x=h, edge_index=edge_index, edge_attr=edge_weight)
            else:
                h = gin_conv(x=h, edge_index=edge_index)
            h = self.batch_norms[layer](h)
            h = self.activation(h)
            hidden_rep.append(h)

        graph_reps = []
        for layer, h_rep in enumerate(hidden_rep):
            h_pooled = self.graph_pool(h_rep, batch)
            graph_reps.append(self.dropout(self.linear_prediction[layer](h_pooled)))

        final_graph_rep = aggregate_tensor(
            torch.stack(graph_reps, dim=-1), aggregation_method=self.layer_aggregation
        )

        return final_graph_rep

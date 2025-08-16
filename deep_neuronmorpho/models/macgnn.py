"""Implementation of Morphology-aware contrastive GNN model (MACGNN) model from [Zhao et al. 2022](https://ieeexplore.ieee.org/document/9895206).

## Model architecture:
- GIN layers for obtaining graph-level rpresentation of each attribute stream
    - Graph pooling (can specify either "mean", "max", or "sum" pooling)
- Two attribute streams (e.g. geometric attributes, topological attributes, etc.)
- Stream aggregation (can specify either "mean", "max", "sum", "wsum", or "cat" aggregation)
    - "wsum" aggregation uses a learnable weight for each stream
- MLP layer processes the aggregated stream features to produce final graph-level embedding
"""

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from deep_neuronmorpho.utils.model_config import GNNConfig

from .gin import GIN
from .mlp import MLP
from .model_utils import (
    aggregate_tensor,
    compute_embedding_dim,
)


class MACGNN(nn.Module):
    """MACGNN model from [Zhao et al. 2022](https://ieeexplore.ieee.org/document/9895206)."""

    def __init__(self, cfg: GNNConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = cfg.hidden_dim
        self.output_dim = self.cfg.output_dim
        self.num_mlp_layers = cfg.num_mlp_layers or 2
        self.num_gnn_layers = cfg.num_gnn_layers
        self.graph_pooling_type = cfg.graph_pooling_type
        self.gnn_layer_aggregation = cfg.gnn_layer_aggregation
        self.dropout_prob = cfg.dropout_prob
        self.stream_aggregation = cfg.stream_aggregation or "mean"
        self.learn_eps = cfg.learn_eps or False
        self.use_edge_weight = cfg.use_edge_weight

        self.geo_gnn = GIN(
            input_dim=3,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.num_gnn_layers,
            num_mlp_layers=self.num_mlp_layers,
            graph_pooling=self.graph_pooling_type,
            layer_aggregation=self.gnn_layer_aggregation,
            dropout_prob=self.dropout_prob,
            learn_eps=self.learn_eps,
        )
        self.topo_gnn = GIN(
            input_dim=5,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.num_gnn_layers,
            num_mlp_layers=self.num_mlp_layers,
            graph_pooling=self.graph_pooling_type,
            layer_aggregation=self.gnn_layer_aggregation,
            dropout_prob=self.dropout_prob,
            learn_eps=self.learn_eps,
        )

        # optional weights for `wsum`
        if self.stream_aggregation == "wsum":
            self.streams_weight = nn.Parameter(
                torch.ones(1, 1, self.num_streams), requires_grad=True
            )
            nn.init.xavier_uniform_(self.streams_weight)
        else:
            self.streams_weight = None

        embedding_dim = compute_embedding_dim(
            hidden_dim=self.hidden_dim,
            num_gnn_layers=self.num_gnn_layers,
            num_streams=self.num_streams,
            gnn_layer_aggregation=self.gnn_layer_aggregation,
            stream_aggregation=self.stream_aggregation,
        )

        self.graph_embedding = MLP(
            input_dim=embedding_dim,
            output_dim=self.output_dim,
            hidden_dim=embedding_dim,
            num_layers=2,
        )

    def process_stream(
        self,
        stream_name: str,
        graphs: Batch,
    ) -> Tensor:
        """Process an attribute stream with its GIN layers.

        Args:
            stream_name (str): Name of the attribute stream in self.attrs_streams dict.
            graphs (Batch): Batch of graphs.
            edge_weight (Tensor | None): Edge weights of the graph. Defaults to None.

        Returns:
            Tensor: Graph-level representation of the attribute stream.
        """
        gnn_stream = self.gnn_streams[stream_name]
        stream_indices = self.attrs_streams[stream_name]

        return gnn_stream(graphs, feat_index=stream_indices)

    def forward(self, graphs: Batch) -> Tensor:
        """Forward pass of the model."""
        # Process each stream
        h_geo = self.geo_gnn(graphs, x=graphs.pos)
        h_topo = self.topo_gnn(graphs, feat_index=list(range(3, 8)))
        # Aggregate streams
        H = torch.stack([h_geo, h_topo], dim=-1)
        H = aggregate_tensor(H, self.stream_aggregation, weights=self.streams_weight)

        return self.graph_embedding(H)

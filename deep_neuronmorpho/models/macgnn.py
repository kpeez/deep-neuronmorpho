"""Implementation of Morphology-aware contrastive GNN model (MACGNN) model from [Zhao et al. 2022](https://ieeexplore.ieee.org/document/9895206).

## Model architecture:
- GIN layers for proecssing graph features
- Graph pooling (can specify either "mean", "max", or "sum" pooling)
- Two or more attribute streams (e.g. geometric attributes, topological attributes, etc.)
- Stream aggregation (can specify either "mean", "max", "sum", "wsum", or "cat" aggregation)
    - "wsum" aggregation uses a learnable weight for each stream
- MLP layer processes the aggregated stream features to produce final graph-level embedding
"""

import torch
from dgl import DGLGraph
from torch import Tensor, nn

from ..utils.model_config import ModelConfig, validate_model_config
from . import GIN
from .model_utils import (
    aggregate_tensor,
    compute_embedding_dim,
    load_attrs_streams,
)


class MACGNN(nn.Module):
    """MACGNN model from [Zhao et al. 2022](https://ieeexplore.ieee.org/document/9895206)."""

    def __init__(self, args: ModelConfig, device: torch.device | None = None) -> None:
        super().__init__()

        self.args = args
        validate_model_config(self.args.to_dict())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_edge_weight = self.args.use_edge_weight
        self.learn_eps = self.args.learn_eps
        self.hidden_dim = self.args.hidden_dim
        self.output_dim = self.args.output_dim
        self.num_mlp_layers = self.args.num_mlp_layers
        self.num_gnn_layers = self.args.num_gnn_layers
        self.graph_pooling_type = self.args.graph_pooling_type
        self.neighbor_aggregation = self.args.neighbor_aggregation
        self.gnn_layer_aggregation = self.args.gnn_layer_aggregation
        self.dropout_prob = self.args.dropout_prob
        self.attrs_streams = load_attrs_streams(self.args.attrs_streams.to_dict())
        self.num_streams = len(self.attrs_streams)
        self.streams_weight = None
        self.stream_aggregation = self.args.stream_aggregation if self.num_streams > 1 else "none"
        self.gnn_streams = nn.ModuleDict()

        # Initialize the GIN layers
        for stream_name, stream_dims in self.attrs_streams.items():
            input_dim = len(stream_dims)
            self.gnn_streams[stream_name] = GIN(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_layers=self.num_gnn_layers,
                num_mlp_layers=self.num_mlp_layers,
                neighbor_aggregation=self.neighbor_aggregation,
                graph_pooling=self.graph_pooling_type,
                layer_aggregation=self.gnn_layer_aggregation,
                dropout_prob=self.dropout_prob,
                learn_eps=self.learn_eps,
            )

        # Initialize stream weights if using weighted sum for streams aggregation
        if self.stream_aggregation == "wsum" and self.num_streams > 1:
            self.streams_weight = nn.Parameter(
                torch.ones(1, 1, self.num_streams), requires_grad=True
            )
            nn.init.xavier_uniform_(self.streams_weight)

        embedding_dim = compute_embedding_dim(
            hidden_dim=self.hidden_dim,
            num_gnn_layers=self.num_gnn_layers,
            num_streams=self.num_streams,
            gnn_layer_aggregation=self.gnn_layer_aggregation,
            stream_aggregation=self.stream_aggregation,
        )

        self.graph_embedding = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
        )

    def process_stream(
        self,
        stream_name: str,
        graphs: DGLGraph,
        h_full: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        """Process an attribute stream with it's GIN layers.

        Args:
            stream_name (str): Name of the attribute stream in self.attrs_streams dict.
            h_full (Tensor): Node features of the graph.
            graphs (DGLGraph): Batch of graph objects from dgl.GraphDataLoader.
            edge_weight (Tensor | None): Edge weights of the graph. Defaults to None.

        Returns:
            Tensor: Graph-level representation of the attribute stream.
        """
        gnn_stream = self.gnn_streams[stream_name]
        stream_indices = self.attrs_streams[stream_name]
        h = h_full[:, stream_indices]

        return gnn_stream(graphs, h, edge_weight)

    def forward(self, graphs: DGLGraph) -> Tensor:
        """Forward pass of the model."""
        h_full = graphs.ndata["nattrs"]
        edge_weight = graphs.edata["edge_weight"] if self.use_edge_weight else None
        h_streams_list = [
            self.process_stream(stream_name, graphs, h_full, edge_weight)
            for stream_name in self.gnn_streams
        ]
        # Aggregate across streams
        if self.stream_aggregation == "none":
            stream_aggregate_graph_rep = h_streams_list[0]

        else:
            h_concat_streams = torch.stack(h_streams_list, dim=-1)
            stream_aggregate_graph_rep = aggregate_tensor(
                h_concat_streams,
                self.stream_aggregation,
                weights=self.streams_weight,  # only used if stream_aggregation == "wsum"
            )

        return self.graph_embedding(stream_aggregate_graph_rep)

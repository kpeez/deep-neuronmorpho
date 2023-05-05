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

from deep_neuronmorpho.models.model_utils import (
    aggregate_tensor,
    compute_embedding_dim,
    create_pooling_layer,
    load_attrs_streams,
)
from deep_neuronmorpho.models.modules import create_gin_layers, linear_block
from deep_neuronmorpho.utils.parse_config import ModelConfig


class MACGNN(nn.Module):
    """MACGNN model from [Zhao et al. 2022](https://ieeexplore.ieee.org/document/9895206)."""

    def __init__(self, args: ModelConfig) -> None:
        super().__init__()

        self.args = args.model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_edge_weight = self.args.use_edge_weight
        self.learn_eps = self.args.learn_eps
        self.hidden_dim = self.args.hidden_dim
        self.output_dim = self.args.output_dim
        self.num_mlp_layers = self.args.num_mlp_layers
        self.num_gnn_layers = self.args.num_gnn_layers
        self.graph_pooling_type = self.args.graph_pooling_type
        self.neighbor_aggregation = self.args.neighbor_aggregation
        self.gnn_layers_aggregation = self.args.gnn_layers_aggregation
        self.stream_aggregation = self.args.stream_aggregation
        self.dropout_prob = self.args.dropout_prob
        self.attrs_streams = load_attrs_streams(self.args.attrs_streams.to_dict())
        self.num_streams = len(self.attrs_streams)
        self.streams_weights = None
        self.gnn_streams = nn.ModuleDict()

        # Initialize the GNN layers
        for stream_name, stream_dims in self.attrs_streams.items():
            input_dim = len(stream_dims)
            self.gnn_streams[stream_name] = create_gin_layers(
                num_gnn_layers=self.num_gnn_layers,
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_mlp_layers=self.num_mlp_layers,
                aggregation_type=self.neighbor_aggregation,
                dropout_prob=self.dropout_prob,
                learn_eps=self.learn_eps,
            )
        self.gpool = create_pooling_layer(self.graph_pooling_type)

        # Initialize stream weights if using weighted sum for streams aggregation
        if self.stream_aggregation == "wsum" and self.num_streams > 1:
            self.streams_weight = nn.Parameter(
                torch.ones(1, 1, self.num_streams), requires_grad=True
            )
            nn.init.xavier_uniform_(self.streams_weight)

    def process_stream(
        self, stream_name: str, h_full: Tensor, graphs: DGLGraph, edge_weight: Tensor | None
    ) -> Tensor:
        """Process an attribute stream.

        Processes an attribute stream with its corresponding GNN layers,
        and aggregate features from all layers in th stream to get graph-level representation.

        Args:
            stream_name (str): Name of the attribute stream in self.attrs_streams dict.
            h_full (Tensor): Node features of the graph.
            graphs (DGLGraph): Batch of graph objects from dgl.GraphDataLoader.
            edge_weight (Tensor | None): Edge weights of the graph. Defaults to None.


        Returns:
            Tensor: Graph-level representation of the attribute stream.
        """
        gnn_layers = self.gnn_streams[stream_name]
        stream_indices = self.attrs_streams[stream_name]
        h = h_full[:, stream_indices]

        h_layers_list = []
        for i in range(self.num_gnn_layers):
            h = gnn_layers[i](h, graphs, edge_weight)
            h_layers_list.append(h)

        h_graph_rep_stream = torch.stack([self.gpool(graphs, h) for h in h_layers_list], dim=-1)
        return aggregate_tensor(h_graph_rep_stream, self.gnn_layers_aggregation)

    def forward(self, graphs: DGLGraph) -> Tensor:
        """Forward pass of the model."""
        h_full = graphs.ndata["nattrs"]
        edge_weight = graphs.edata["edge_weight"] if self.use_edge_weight else None

        h_streams_list = []
        for stream_name in self.gnn_streams:
            h_graph_rep_stream = self.process_stream(stream_name, h_full, graphs, edge_weight)
            h_streams_list.append(h_graph_rep_stream)

        # Aggregate across streams
        h_concat_streams = torch.stack(h_streams_list, dim=-1)
        stream_aggregate_graph_rep = aggregate_tensor(
            h_concat_streams,
            self.stream_aggregation,
            weights=self.streams_weight,  # only used if stream_aggregation == "wsum"
        )
        embedding_dim = compute_embedding_dim(
            hidden_dim=self.hidden_dim,
            num_gnn_layers=self.num_gnn_layers,
            num_streams=self.num_streams,
            gnn_layers_aggregation=self.gnn_layers_aggregation,
            stream_aggregation=self.stream_aggregation,
        )
        final_graph_rep = stream_aggregate_graph_rep
        final_graph_rep = linear_block(
            input_dim=embedding_dim,
            output_dim=self.output_dim,
        )(stream_aggregate_graph_rep)

        return final_graph_rep

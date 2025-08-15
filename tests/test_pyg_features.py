from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx

from deep_neuronmorpho.data import SWCData, compute_neuron_node_feats, swc_df_to_pyg_data


def create_graph_morphopy(swc_file: str | Path) -> nx.DiGraph:
    """Create networkx graph of neuron from swc format using MorphoPy for features.

    This function is deprecated in favor of `convert_swc_to_pyg`. It is only used for testing
    the `convert_swc_to_pyg` function.

    Args:
        swc_file (str | Path): Morphopy NeuronTree object of neuron swc data.

    Returns:
        nx.DiGraph: Graph of neuron.

    Notes:
        This function takes in a MorphoPy NeuronTree object and returns a networkx graph with the
        following node attributes:

            1. x: x coordinate of node.
            2. y: y coordinate of node.
            3. z: z coordinate of node.
            4. euclidean_dist_norm: normalized euclidean distance from soma.
            5. path_dist_norm: normalized path distance from soma.
            6. tortuosity: tortuosity of the path.
            7. branch_order: branch order of the node.
            8. strahler_order: strahler order of the node.
    """
    neuron_tree = SWCData(swc_file=swc_file, standardize=False, align=False).ntree
    euclidean_dist = neuron_tree.get_radial_distance()
    path_dist = neuron_tree.get_path_length(weight="path_length")
    euclidean_dist_norm = {k: np.log1p(v) for k, v in euclidean_dist.items()}
    path_dist_norm = {k: np.log1p(v) for k, v in path_dist.items()}
    tortuosity = {k: v / max(euclidean_dist[k], 1e-12) for k, v in path_dist.items()}
    # set root tortuosity to 1.0 (undefined at soma)
    root_id = min(euclidean_dist, key=euclidean_dist.get)
    if euclidean_dist[root_id] <= 1e-12:
        tortuosity[root_id] = 1.0

    branch_order = neuron_tree.get_branch_order()
    strahler_order = neuron_tree.get_strahler_order()

    neuron_graph = neuron_tree.get_graph()

    for node in neuron_graph.nodes():
        node_feats = np.array(
            [
                *neuron_graph.nodes[node]["pos"],
                euclidean_dist_norm[node],
                path_dist_norm[node],
                tortuosity[node],
                branch_order[node],
                strahler_order[node],
            ],
            dtype=np.float32,
        )
        neuron_graph.nodes[node].clear()
        neuron_graph.nodes[node]["feats"] = node_feats

    # morphopy uses 1-indexing for nodes, we should use 0-indexing
    neuron_graph = nx.relabel_nodes(neuron_graph, {i: i - 1 for i in neuron_graph.nodes()})

    for _, _, edge_data in neuron_graph.edges(data=True):
        del edge_data["euclidean_dist"]
        del edge_data["path_length"]

    return neuron_graph


def test_pyg_features_against_morphopy(swc_file: Path):
    """Test that the features computed by the `convert_swc_to_pyg` function are the same as the features computed by MorphoPy."""
    morphopy_graph = create_graph_morphopy(swc_file)
    morphopy_pyg = from_networkx(morphopy_graph, group_node_attrs="feats")
    morphopy_pyg_feats = morphopy_pyg.x.to(torch.float32)

    df = SWCData.load_swc_data(swc_file)
    pyg_graph = swc_df_to_pyg_data(df)
    pyg_features = compute_neuron_node_feats(pyg_graph.x, pyg_graph.edge_index, pyg_graph.root)

    assert torch.allclose(morphopy_pyg_feats, pyg_features)

"""Prepare neuron graphs for conversion to graph datasets."""

from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean

from .process_swc import SWCData
from .utils import compute_edge_weights, compute_graph_attrs


def create_neuron_graph(swc_file: str | Path) -> nx.DiGraph:
    """Create networkx graph of neuron from swc format.

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
            4. r: node radius. (not currently in use due to unreliable radius data)
            5. path_dist: path distance from soma.
            6. euclidean_dist: euclidean distance from soma.
            7.-12. angle attrs (n=6): min, mean, median, max, std, num of branch angles.
            13.-18. branch attrs (n=6): min, mean, median, max, std, num of branch lengths.
    """
    neuron_tree = SWCData(swc_file=swc_file, standardize=False, align=False).ntree
    angles = list(neuron_tree.get_branch_angles().values())
    branches = list(neuron_tree.get_segment_length().values())
    angle_stats = compute_graph_attrs(angles)
    branch_stats = compute_graph_attrs(branches)
    neuron_graph = neuron_tree.get_graph()
    # morphopy uses 1-indexing for nodes, we should use 0-indexing
    neuron_graph = nx.relabel_nodes(neuron_graph, {i: i - 1 for i in neuron_graph.nodes()})
    soma_x, soma_y, soma_z = neuron_graph.nodes[1]["pos"]
    for node in neuron_graph.nodes():
        x, y, z = neuron_graph.nodes[node]["pos"]
        node_attrs = [
            x,
            y,
            z,
            # neuron_graph.nodes[node]["radius"],  # uncomment to include radius
            euclidean((x, y, z), (soma_x, soma_y, soma_z)),
            nx.dijkstra_path_length(neuron_graph, 0, node, weight="path_length"),
            *angle_stats,
            *branch_stats,
        ]
        neuron_graph.nodes[node].clear()
        neuron_graph.nodes[node].update({"nattrs": [np.float32(attr) for attr in node_attrs]})

    for _, _, edge_data in neuron_graph.edges(data=True):
        del edge_data["euclidean_dist"]
        del edge_data["path_length"]
    # get path_idx depending on if we have radius in dataset or not
    path_idx = 5 if len(neuron_graph.nodes(data=True)[0]["nattrs"]) == 18 else 4
    neuron_graph = compute_edge_weights(neuron_graph, path_idx=path_idx)

    return neuron_graph

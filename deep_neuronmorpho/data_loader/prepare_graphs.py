"""Prepare neuron graphs for conversion to DGL datasets."""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from morphopy.neurontree import NeuronTree as nt
from scipy import stats
from scipy.spatial.distance import euclidean


def load_swc_file(swc_file: Path | str) -> pd.DataFrame:
    """Load swc file.

    Args:
        swc_file (Path): Path to swc file.

    Returns:
        pd.DataFrame: SWC data.
    """
    swc_data = pd.read_csv(
        swc_file,
        sep=" ",
        header=None,
        names=["n", "type", "x", "y", "z", "radius", "parent"],
    )

    if swc_data[["x", "y", "z"]].iloc[0].all() != 0.0:
        x, y, z = swc_data[["x", "y", "z"]].iloc[0]
        swc_data[["x", "y", "z"]] = swc_data[["x", "y", "z"]] - [x, y, z]

    return swc_data


def load_neuron_tree(
    neuron_swc: pd.DataFrame,
    resample_dist: int | float | None = None,
) -> nt.NeuronTree:
    """Load NeuronTree from MorphoPy.

    Args:
        neuron_swc (pd.DataFrame): swc data.
        resample_dist (float, optional): Resample distance in microns. Defaults to None.

    Returns:
        NeuronTree: NeuronTree object.
    """
    if "id" in neuron_swc.columns:
        neuron_swc.rename(columns={"id": "n"}, inplace=True)

    neuron_tree = nt.NeuronTree(neuron_swc)

    if resample_dist:
        neuron_tree = neuron_tree.resample_tree(resample_dist)

    return neuron_tree


def compute_graph_attrs(graph_attrs: list[float]) -> list[float]:
    """Compute summary statistics for a list of graph attributes.

    Args:
        graph_attrs (list[float]): Graph attribute data.

    Returns:
        dict[str, float]: Summary statistics of graph attributes. In the following order:
         min, mean, median, max, std, num of observations

    """
    res = stats.describe(graph_attrs)
    attr_stats = [
        res.minmax[0],
        res.mean,
        np.median(graph_attrs),
        res.minmax[1],
        (res.variance**0.5),
        res.nobs,
    ]
    return attr_stats


def create_neuron_graph(swc_file: str | Path, resample_dist: int | float | None = None) -> nx.Graph:
    """Create networkx graph of neuron from swc format.

    Args:
        swc_file (str | Path): Morphopy NeuronTree object of neuron swc data.
        resample_dist (int | float, optional): Resample distance in microns. Defaults to None.

    Returns:
        nx.Graph: Graph of neuron.

    Notes:
        This function takes in a MorphoPy NeuronTree object and returns a networkx graph with the
        following node attributes:

            1. x: x coordinate of node.
            2. y: y coordinate of node.
            3. z: z coordinate of node.
            4. r: node radius.
            5. path_dist: path distance from soma.
            6. euclidean_dist: euclidean distance from soma.
            7.-12. angle attrs (n=6): min, mean, median, max, std, num of branch angles.
            13.-18. branch attrs (n=6): min, mean, median, max, std, num of branch lengths.
    """
    swc_data = load_swc_file(swc_file)
    neuron_tree = load_neuron_tree(swc_data, resample_dist=resample_dist)
    # get branch angles and branch lengths
    angles = list(neuron_tree.get_branch_angles().values())
    branches = list(neuron_tree.get_segment_length().values())
    angle_stats = compute_graph_attrs(angles)
    branch_stats = compute_graph_attrs(branches)
    # update graph attributes
    neuron_graph = neuron_tree.get_graph()
    soma_x, soma_y, soma_z = neuron_graph.nodes[1]["pos"]
    for node in neuron_graph.nodes():
        # expand position to x, y, z
        x, y, z = neuron_graph.nodes[node]["pos"]
        radius = neuron_graph.nodes[node]["radius"]
        node_attrs = (
            [
                x,
                y,
                z,
                radius,
                nx.dijkstra_path_length(neuron_graph, 1, node, weight="path_length"),
                euclidean((x, y, z), (soma_x, soma_y, soma_z)),
            ]
            + angle_stats
            + branch_stats
        )

        neuron_graph.nodes[node].clear()
        neuron_graph.nodes[node].update({"nattrs": [np.float32(attr) for attr in node_attrs]})
    # euclidean_dist is duplicate of path_dist for edges so remove
    # TODO: fix to compute "attention" instead
    for _, _, edge_data in neuron_graph.edges(data=True):
        del edge_data["euclidean_dist"]

    return neuron_graph

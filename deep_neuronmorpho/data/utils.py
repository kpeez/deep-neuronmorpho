"""Create training, validation, and test splits of the dataset."""

from collections.abc import Sequence

import networkx as nx
import numpy as np
from scipy import stats


def compute_edge_weights(G: nx.DiGraph, path_idx: int, epsilon: float = 1.0) -> nx.DiGraph:
    """Compute edge attention weights for a graph.

    Based on the method described in [Zhao et al. 2022](https://ieeexplore.ieee.org/document/9895206).

    Args:
        G (nx.DiGraph): Graph to compute edge weights for.
        path_idx (int): Index of path distance in node attributes.
        epsilon (float, optional): Small constant to prevent division by zero. Defaults to 1.0.

    Returns:
        nx.DiGraph: Graph with attention weights added as edge attributes.
    """
    for u, v in G.edges:
        G.add_edge(v, u)

    node_coeffs = {
        node: 1.0 / (ndata["nattrs"][path_idx] + epsilon) for node, ndata in G.nodes(data=True)
    }
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        local_coeffs = np.array([node_coeffs[neighbor] for neighbor in neighbors], dtype=np.float32)
        attention_coeffs = np.exp(local_coeffs) / np.sum(np.exp(local_coeffs))

        edge_weights = (
            (neighbor, coeff) for neighbor, coeff in zip(neighbors, attention_coeffs, strict=True)
        )
        for neighbor, weight in edge_weights:
            G[node][neighbor]["edge_weight"] = weight

    return G


def compute_graph_attrs(graph_attrs: Sequence[float]) -> list[float]:
    """Compute summary statistics for a list of graph attributes.

    Args:
        graph_attrs (Sequence[float]): Graph attribute data.

    Returns:
        list[float]: Summary statistics of graph attributes. In the following order:
         min, mean, median, max, std, num of observations

    """
    # some graphs don't have attributes but we don't want to break them downstream
    if not graph_attrs:
        graph_attrs.extend([0, 0])

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

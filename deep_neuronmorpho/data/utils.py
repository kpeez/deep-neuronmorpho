"""Create training, validation, and test splits of the dataset."""

from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .process_swc import SWCData


def convert_swc_to_pyg(
    df: pd.DataFrame,
    *,
    restrict_to_root_component: bool = True,
    include_traversal_indices: bool = True,
    assume_first_row_root: bool = False,
) -> Data:
    """
    Build a PyG Data from an SWC-like DataFrame (columns: n, type, x, y, z, radius, parent),
    using DFS pre-order as the canonical node order (soma/root becomes row 0).

    Args:
        df: SWC-like DataFrame containing at least columns: {"n", "parent", "x", "y", "z"}.
        restrict_to_root_component: Currently only the soma-connected component is returned.
            Reserved for future use; must remain True.
        include_traversal_indices: If True, attach per-row DFS entry/exit indices
            (data.dfs_first, data.dfs_last). Defaults to True.
        assume_first_row_root: If True, assume the first row's id is the soma/root id.
            Otherwise, infer root via parent == -1 (fallback to type == 1 or first row).

    Returns:
        Data with attributes:
          - x: [N, 3] float32 positions in DFS pre-order
          - edge_index: [2, E] long parent->child (restricted to soma component)
          - root: int (0)
          - orig_id: [N] long original SWC 'n' in DFS order
          - dfs_first/dfs_last: [N] long entry/exit indices (included if requested)
    """
    if not restrict_to_root_component:
        # Avoid silent changes of behavior; current implementation only returns the root component.
        raise NotImplementedError("restrict_to_root_component=False is not supported yet")
    required = {"n", "parent", "x", "y", "z"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"SWC DataFrame is missing required columns: {sorted(missing)}")

    n = df["n"].to_numpy(np.int64)
    parent = df["parent"].to_numpy(np.int64)
    xyz = df[["x", "y", "z"]].to_numpy(np.float32)
    soma_id = int(n[0]) if assume_first_row_root else compute_soma_id(n, parent, df)
    neigh = build_undirected_neighbors(n, parent)
    dfs_ids, first_id, last_id = dfs_preorder(neigh, soma_id)
    # map ids -> original file row for fast vectorized gather
    id2row = {int(_n): i for i, _n in enumerate(n)}
    idx_orig = np.fromiter((id2row[i] for i in dfs_ids), dtype=np.int64, count=len(dfs_ids))
    pos = torch.from_numpy(xyz[idx_orig])  # [N,3], float32
    orig_id = torch.as_tensor(dfs_ids, dtype=torch.long)  # [N], original SWC ids
    id_in_comp = set(dfs_ids)
    new_index = {nid: i for i, nid in enumerate(dfs_ids)}

    mask = parent != -1
    child_ids = n[mask]
    parent_ids = parent[mask]

    edge_par, edge_ch = [], []
    for cid_raw, pid_raw in zip(child_ids, parent_ids, strict=True):
        cid = int(cid_raw)
        pid = int(pid_raw)
        if cid in id_in_comp and pid in id_in_comp:
            edge_par.append(new_index[pid])
            edge_ch.append(new_index[cid])

    edge_index = torch.tensor([edge_par, edge_ch], dtype=torch.long)
    data = Data(x=pos, edge_index=edge_index)
    data.root = 0
    data.orig_id = orig_id
    if include_traversal_indices:
        dfs_first = torch.empty(len(dfs_ids), dtype=torch.long)
        dfs_last = torch.empty(len(dfs_ids), dtype=torch.long)
        for nid, row_idx in new_index.items():
            dfs_first[row_idx] = first_id[nid]
            dfs_last[row_idx] = last_id[nid]
        data.dfs_first = dfs_first
        data.dfs_last = dfs_last
    # alias coordinates to PyG position
    data.pos = data.x

    return data


def compute_soma_id(n: np.ndarray, parent: np.ndarray, df: pd.DataFrame) -> int:
    """
    Determine the soma/root SWC id from arrays and dataframe.

    Preference order:
      1) parent == -1 (SWC root definition)
      2) if none and 'type' column exists: type == 1 (soma type)
      3) fallback: first row's id
    """
    soma_mask = parent == -1
    if soma_mask.sum() == 0 and "type" in df.columns:
        soma_mask = df["type"].to_numpy(np.int64) == 1
    if soma_mask.sum() == 0:
        return int(n[0])
    return int(n[soma_mask][0])


def build_undirected_neighbors(n: np.ndarray, parent: np.ndarray) -> dict[int, list[int]]:
    """
    Build an undirected adjacency mapping on SWC ids.
    Guards against stray parent references not present in `n`.
    """
    neighbors: dict[int, list[int]] = defaultdict(list)
    id_set = {int(_n) for _n in n}
    for cid_raw, pid_raw in zip(n, parent, strict=False):
        cid = int(cid_raw)
        pid = int(pid_raw)
        if pid == -1:
            continue
        if pid in id_set:
            neighbors[pid].append(cid)
            neighbors[cid].append(pid)
    # ensure keys for all ids (even isolated)
    for _n in n:
        neighbors.setdefault(int(_n), [])
    return dict(neighbors)


def dfs_preorder(
    neighbors: dict[int, list[int]], root_id: int
) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """
    Iterative DFS pre-order over ids starting at `root_id`.

    Returns:
      - dfs_ids: list of node ids in DFS pre-order (root first)
      - entry_idx: id -> DFS entry index
      - exit_idx:  id -> DFS exit  index (inclusive)
    Only visits the root-connected component.
    """
    dfs_ids: list[int] = []
    entry_idx: dict[int, int] = {}
    exit_idx: dict[int, int] = {}
    stack: list[tuple[int, int, int]] = [(root_id, root_id, 0)]
    seen = {root_id}
    while stack:
        u, p, state = stack.pop()
        if state == 0:
            entry_idx[u] = len(dfs_ids)
            dfs_ids.append(u)
            stack.append((u, p, 1))
            # push children in reverse for natural order; skip parent edge
            for v in reversed(neighbors.get(u, [])):
                if v == p or v in seen:
                    continue
                seen.add(v)
                stack.append((v, u, 0))
        else:
            exit_idx[u] = len(dfs_ids) - 1

    return dfs_ids, entry_idx, exit_idx


def create_neuron_graph(swc_file: str | Path) -> nx.DiGraph:
    """Create networkx graph of neuron from swc format.

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
            4. r: node radius. (not currently in use due to unreliable radius data)
            5. path_dist: path distance from soma.
            6. euclidean_dist: euclidean distance from soma.
            7.-12. angle attrs (n=6): min, mean, median, max, std, num of branch angles.
            13.-18. branch attrs (n=6): min, mean, median, max, std, num of branch lengths.
    """
    neuron_tree = SWCData(swc_file=swc_file, standardize=False, align=False).ntree
    euclidean_dist = neuron_tree.get_radial_distance()
    path_dist = neuron_tree.get_path_length(weight="path_length")
    euclidean_dist_norm = {k: v / max(euclidean_dist.values()) for k, v in euclidean_dist.items()}
    path_dist_norm = {k: v / max(path_dist.values()) for k, v in path_dist.items()}
    tortuosity = {k: v / max(euclidean_dist_norm[k], 1e-8) for k, v in path_dist_norm.items()}
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


def compute_edge_weights(G: nx.DiGraph, epsilon: float = 1.0) -> nx.DiGraph:
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

    node_coeffs = {node: 1.0 / (ndata["nattrs"][4] + epsilon) for node, ndata in G.nodes(data=True)}
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

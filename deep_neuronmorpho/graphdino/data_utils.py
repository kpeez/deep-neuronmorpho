"""Create training, validation, and test splits of the dataset."""

from collections import deque

import numpy as np
import torch


def find_leaf_nodes(neighbors: dict[int, list[int]]) -> list[int]:
    """
    Create list of candidates for leaf and branching nodes.
    Identifies all leaf nodes (degree 1) and follows branch paths (degree 2)
    to create a collection of all terminal structures in the neuron.

    Args:
        neighbors: dict of neighbors per node

    Returns:
        List of node IDs representing leaf nodes and their connecting pathways
    """
    node_degrees = {node: len(neighbors[node]) for node in neighbors}
    leafs = [node for node, degree in node_degrees.items() if degree == 1]
    candidates = set(leafs)
    # find all nodes with degree 2 that are not in candidates
    next_nodes = deque()
    for leaf in leafs:
        for neighbor in neighbors[leaf]:
            if node_degrees[neighbor] == 2 and neighbor not in candidates:
                next_nodes.append(neighbor)

    while next_nodes:
        current = next_nodes.popleft()
        candidates.add(current)

        for neighbor in neighbors[current]:
            if (
                node_degrees[neighbor] == 2
                and neighbor not in candidates
                and neighbor not in next_nodes
            ):
                next_nodes.append(neighbor)

    return list(candidates)


def compute_path_lengths(source_idx: int, neighbors: dict[int, list[int]]) -> dict[int, int]:
    """
    Computes shortest path distances from a source node to all reachable nodes.
    Uses breadth-first search to find the minimum number of edges between nodes.

    Args:
        source_idx: Index of the source node
        neighbors: Dictionary mapping node indices to their neighbor indices

    Returns:
        Dictionary mapping node indices to their distance from the source node
    """
    queue = [source_idx]
    distances = {source_idx: 0}

    while queue:
        current = queue.pop(0)
        current_dist = distances[current]

        for neighbor in neighbors[current]:
            if neighbor not in distances:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

    return distances


def subsample_graph(
    neighbors: dict[int, list[int]] | None = None,
    not_deleted: list[int] | None = None,
    keep_nodes: int = 200,
    protected: list[int] | None = None,
) -> tuple[dict[int, list[int]], list[int]]:
    """
    Subsample graph to a fixed number of nodes.

    Args:
        neighbors: dict of neighbors per node
        not_deleted: list of nodes, who did not get deleted in previous processing steps
        keep_nodes: number of nodes to keep in graph
        protected: nodes to be excluded from subsampling

    Returns:
        tuple of (neighbors, not_deleted)
    """
    if neighbors is not None:
        k_nodes = len(neighbors)
    else:
        raise ValueError("neighbors must be provided")

    if protected is None:
        protected = [0]

    # protect soma node from being removed
    protected = set(protected)

    # Set fixed seed for reproducibility
    torch.manual_seed(42)

    # indices as set in random order
    perm = torch.randperm(k_nodes).tolist()
    all_indices = np.array(list(not_deleted))[perm].tolist()
    deleted = set()

    while len(deleted) < k_nodes - keep_nodes:
        while True:
            if len(all_indices) == 0:
                assert len(not_deleted) > keep_nodes, len(not_deleted)
                remaining = list(not_deleted - deleted)
                torch.manual_seed(42)  # Reset seed for consistency
                perm = torch.randperm(len(remaining)).tolist()
                all_indices = np.array(remaining)[perm].tolist()

            idx = all_indices.pop()

            if idx not in deleted and len(neighbors[idx]) < 3 and idx not in protected:
                break

        if len(neighbors[idx]) == 2:
            n1, n2 = neighbors[idx]
            neighbors[n1].remove(idx)
            neighbors[n2].remove(idx)
            neighbors[n1].add(n2)
            neighbors[n2].add(n1)
        elif len(neighbors[idx]) == 1:
            n1 = neighbors[idx].pop()
            neighbors[n1].remove(idx)

        del neighbors[idx]
        deleted.add(idx)

    not_deleted = list(not_deleted - deleted)
    return neighbors, not_deleted


def drop_random_branch(
    nodes: list[int],
    neighbors: dict[int, list[int]],
    distances: dict[int, int],
    keep_nodes: int = 200,
) -> tuple[dict[int, list[int]], set[int]]:
    """
    Removes a terminal branch. Starting nodes should be between
    branching node and leaf (see leaf_branch_nodes)

    Args:
        nodes: List of nodes of the graph
        neighbors: Dict of neighbors per node
        distances: Dict of distances of nodes to origin
        keep_nodes: Number of nodes to keep in graph
    """
    start = list(nodes)[torch.randint(len(nodes), (1,)).item()]
    to = next(iter(neighbors[start]))

    if distances[start] > distances[to]:
        start, to = to, start

    drop_nodes = [to]
    next_nodes = [n for n in neighbors[to] if n != start]

    while next_nodes:
        s = next_nodes.pop(0)
        drop_nodes.append(s)
        next_nodes += [n for n in neighbors[s] if n not in drop_nodes]

    if len(neighbors) - len(drop_nodes) < keep_nodes:
        return neighbors, set()
    else:
        # Delete nodes.
        for key in drop_nodes:
            if key in neighbors:
                for k in neighbors[key]:
                    neighbors[k].remove(key)
                del neighbors[key]

        return neighbors, set(drop_nodes)


def remap_neighbors(neighbors: dict[int, list[int]]) -> tuple[dict[int, list[int]], dict[int, int]]:
    """
    Remap node indices to be between 0 and the number of nodes.

    Args:
        neighbors: Dict of node id mapping to the node's neighbors.
    Returns:
        ordered_neighbors: Dict with neighbors with new node ids.
        subsampled2new: Mapping between old and new indices (dict).
    """
    # Create maps between new and old indices.
    subsampled2new = {k: i for i, k in enumerate(sorted(neighbors))}

    # Re-map indices to 1..N.
    ordered_neighbors = {i: neighbors[k] for i, k in enumerate(sorted(neighbors))}

    # Re-map keys of neighbors
    for k, v in ordered_neighbors.items():
        ordered_neighbors[k] = {subsampled2new[x] for x in v}

    return ordered_neighbors, subsampled2new


def neighbors_to_adjacency_torch(
    neighbors: dict[int, list[int]], not_deleted: list[int]
) -> torch.Tensor:
    """Create adjacency matrix from list of non-empty neighbors."""
    node_map = {n: i for i, n in enumerate(not_deleted)}

    n_nodes = len(not_deleted)

    new_adj_matrix = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    for ii, nodes in neighbors.items():
        for jj in nodes:
            i, j = node_map[ii], node_map[jj]
            new_adj_matrix[i, i] = True  # diagonal if needed
            new_adj_matrix[i, j] = True
            new_adj_matrix[j, i] = True

    return new_adj_matrix


def compute_laplacian_eigenvectors(adj_matrix: torch.Tensor, pos_enc_dim: int = 32) -> torch.Tensor:
    """Compute positional encoding using graph laplacian.
        Adapted from https://github.com/graphdeeplearning/benchmarking-gnns/blob/ef8bd8c7d2c87948bc1bdd44099a52036e715cd0/data/molecules.py#L147-L168.

    Args:
        adj_matrix: Adjacency matrix (B x N x N).
        pos_enc_dim: Output dimensions of positional encoding.
    """
    b, n, _ = adj_matrix.size()
    # laplacian
    A = adj_matrix.float()
    degree_matrix = A.sum(axis=1).clip(1)
    N = torch.diag_embed(degree_matrix**-0.5)
    L = torch.eye(n, device=A.device)[None,].repeat(b, 1, 1) - (N @ A) @ N
    # eigenvectors
    _, eig_vec = torch.linalg.eigh(L)
    eig_vec = torch.flip(eig_vec, dims=[2])
    pos_enc = eig_vec[:, :, 1 : pos_enc_dim + 1]

    if pos_enc.size(2) < pos_enc_dim:
        pos_enc = torch.cat(
            [
                pos_enc,
                torch.zeros(
                    pos_enc.size(0), pos_enc.size(1), pos_enc_dim - pos_enc.size(2), device=A.device
                ),
            ],
            dim=2,
        )

    return pos_enc

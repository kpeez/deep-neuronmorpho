import torch
from torch import Tensor
from torch_geometric.utils import to_undirected

FEATURE_NAMES = [
    "x",
    "y",
    "z",
    "radial_log",
    "path_log",
    "tortuosity",
    "branch_order",
    "strahler_order",
]


@torch.no_grad()
def compute_neuron_node_feats(pos: Tensor, edge_index: Tensor, root: int) -> Tensor:
    """
    This function computes the following features for a neuron:

        1. x: x coordinate of node.
        2. y: y coordinate of node.
        3. z: z coordinate of node.
        4. radial_log: log of radial distance from soma.
        5. path_log: log of path distance from soma.
        6. tortuosity: tortuosity of the path.
        7. branch_order: branch order of the node.
        8. strahler_order: strahler order of the node.

    Args:
        pos: [N,3] float32 positions in DFS pre-order
        edge_index: [2,E] long parent->child (restricted to soma component)
        root: int

    Returns [N,8]: [x, y, z, radial_log, path_log, tortuosity, branch_order, strahler_order]
    """
    N = pos.size(0)
    parent, order, visited = dfs_orient_undirected(
        edge_index=edge_index, root=int(root), num_nodes=N
    )
    if (~visited).any():
        raise ValueError("Graph contains unreachable/isolated nodes; fix construction.")

    radial_log, path_log, tortuosity = compute_geometry_features(pos, parent, order, visited, root)
    branch_order, strahler_order = compute_topology_features(parent, order, N)

    x = torch.stack(
        [
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            radial_log,
            path_log,
            tortuosity,
            branch_order.to(pos.dtype),
            strahler_order.to(pos.dtype),
        ],
        dim=1,
    )
    if (~visited).any():
        x[~visited] = 0

    return x


# 1) Orientation (DFS): parent, order, visited
def dfs_orient_undirected(
    edge_index: Tensor, root: int, num_nodes: int
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Build DFS pre-order from `root` on the undirected view.

    Args:
        edge_index: [2,E] long parent->child (restricted to soma component)
        root: int
        num_nodes: int

    Returns:
      parent:  [N] long  (parent[root]=root, -1 for unreachable)
      order:   [M] long  (DFS order over soma component; M <= N)
      visited: [N] bool
    """

    ei = to_undirected(edge_index, num_nodes=num_nodes)
    adj = [[] for _ in range(num_nodes)]
    for u, v in ei.t().tolist():
        adj[u].append(v)
        adj[v].append(u)

    parent = torch.full((num_nodes,), -1, dtype=torch.long)
    order = []
    stack = [int(root)]
    parent[root] = root
    while stack:
        u = stack.pop()
        order.append(u)
        for v in reversed(adj[u]):  # reversed for stable child order
            if parent[v] == -1:
                parent[v] = u
                stack.append(v)

    visited = torch.zeros(num_nodes, dtype=torch.bool)
    visited[order] = True
    return parent, torch.tensor(order, dtype=torch.long), visited


def compute_geometry_features(
    pos: Tensor,
    parent: Tensor,
    order: Tensor,
    visited: Tensor,
    root: int,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute radial (euclidean) and path distances from soma, and tortuosity.

    Note: We use log-transformed distances to preserve global scale while avoiding numerical instability.

    Computes:
        - radial_log: log of radial distance from soma
        - path_log: log of path distance from soma
        - tortuosity: tortuosity of the path

    Args:
        pos: [N,3] float32 positions in DFS pre-order
        parent: [N] long parent[root]=root, -1 for unreachable
        order: [M] long DFS order over soma component; M <= N
        visited: [N] bool
        root: int

    Returns:
      radial_log, path_log, tortuosity
    """
    device, dtype = pos.device, pos.dtype
    N = pos.size(0)
    idx = torch.arange(N, device=device)

    # edge length to parent
    elen = torch.zeros(N, device=device, dtype=dtype)
    mask = visited & (parent != idx)
    ch = idx[mask]
    if ch.numel():
        elen[ch] = (pos[ch] - pos[parent[mask]]).norm(dim=1)

    # path (top-down; order guarantees parent before child)
    path = torch.zeros(N, device=device, dtype=dtype)
    for v in order[1:]:
        p = parent[v]
        path[v] = path[p] + elen[v]

    # radial from soma
    radial = torch.zeros(N, device=device, dtype=dtype)
    radial[visited] = (pos[visited] - pos[int(root)]).norm(dim=1)

    tortuosity = torch.ones(N, device=device, dtype=dtype)
    denom = radial.clamp_min(eps)
    mask_div = visited & (radial > eps)
    tortuosity[mask_div] = path[mask_div] / denom[mask_div]
    tortuosity[int(root)] = torch.as_tensor(1.0, device=device, dtype=dtype)

    return torch.log1p(radial), torch.log1p(path), tortuosity


def compute_topology_features(
    parent: Tensor, order: Tensor, num_nodes: int
) -> tuple[Tensor, Tensor]:
    """
    Compute branch and Strahler orders from orientation.

    Branch order is the number of branch points on the path to the node (top-down).
    Strahler order is the maximum order of the children of the node (bottom-up).

    Args:
        parent: [N] long parent[root]=root, -1 for unreachable
        order: [M] long DFS order over soma component; M <= N
        num_nodes: int

    Returns:
      branch_order [N] long, strahler [N] long
    """
    children = [[] for _ in range(num_nodes)]
    for v in order[1:]:
        p = parent[v]
        children[p].append(int(v))

    device = order.device
    outdeg = torch.tensor(
        [len(children[u]) for u in range(num_nodes)], dtype=torch.long, device=device
    )
    is_branchpoint = (outdeg >= 2).to(torch.long)
    # branch order (increment if parent is a branch point)
    branch_order = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for u in order:
        for v in children[int(u)]:
            branch_order[v] = branch_order[u] + is_branchpoint[u]
    # Strahler (bottom-up)
    strahler_order = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for u in reversed(order.tolist()):
        child = children[u]
        if not child:
            strahler_order[u] = 1
        else:
            Sv = strahler_order[torch.tensor(child, dtype=torch.long, device=device)]
            m = int(Sv.max().item())
            strahler_order[u] = m + 1 if int((Sv == m).sum().item()) >= 2 else m

    return branch_order, strahler_order

"""Tensor-based augmentations of neuron structures for contrastive learning."""

import numpy as np
import torch
from torch.distributions.uniform import Uniform


def jitter_node_positions(
    node_features: torch.Tensor,
    jitter: float,
) -> torch.Tensor:
    """Shift node positions by adding Gaussian noise.

    This augmentation shifts 3D coordinates of all points by adding
    Gaussian noise to their positions, similar to PyTorch Geometric's RandomJitter.

    Args:
        node_features: Tensor of shape (num_nodes, feat_dim) containing node features.
                       XYZ coordinates are assumed to be in the first 3 columns.
        std_noise: Standard deviation of Gaussian noise.

    Returns:
        Augmented node features tensor.
    """
    node_features = node_features.clone()
    node_features[:, :3] += torch.normal(
        mean=0,
        std=jitter,
        size=(node_features.shape[0], 3),
        device=node_features.device,
    )
    return node_features


def translate_all_nodes(node_features: torch.Tensor, translate_var: float) -> torch.Tensor:
    """Translate all nodes by a specified amount.

    This augmentation shifts 3D coordinates of all points by adding a constant vector,
    similar to PyTorch Geometric's RandomTranslate.

    Args:
        node_features: Tensor of shape (num_nodes, feat_dim) containing node features.
                       XYZ coordinates are assumed to be in the first 3 columns.
        translate_var: Amount to translate each node.

    Returns:
        Augmented node features tensor.
    """
    jitter = torch.randn(3, device=node_features.device, dtype=torch.float32) * translate_var
    node_features[:, :3] += jitter

    return node_features


def rotate_node_positions(node_features: torch.Tensor, axis: str | None = None) -> torch.Tensor:
    """Perform a random 3D rotation on node coordinates.

    This augmentation rotates the input tensor along the given axis, using [Rodrigues' rotation formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula).

    Args:
        node_features: Tensor of shape (num_nodes, feat_dim) containing node features.
                       XYZ coordinates are assumed to be in the first 3 columns.

    Returns:
        Augmented node features tensor with rotated XYZ coordinates.
    """
    if axis is None:
        return node_features

    device = node_features.device

    if axis is not None:
        if axis == "x":
            rotation_axis = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)
        elif axis == "y":
            rotation_axis = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
        elif axis == "z":
            rotation_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)

    # Generate rotation angle
    angle_dist = Uniform(0, np.pi)
    theta = angle_dist.sample().to(device)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

    # Orthonormal unit vector along rotation axis
    u = rotation_axis / rotation_axis.norm()

    # Outer product of u with itself used to project vectors onto the plane perpendicular to u
    outer = torch.ger(u, u)

    # This matrix rotates vectors along `u` axis by angle `theta`
    rotate_mat = (
        cos_theta * torch.eye(3, device=device)  # Rotation about rotate_axis
        + sin_theta
        * torch.tensor(  # Rotation about plane perpendicular to rotate_axis
            [
                [0, -u[2], u[1]],
                [u[2], 0, -u[0]],
                [-u[1], u[0], 0],
            ],
            device=device,
        )
        + (1 - cos_theta) * outer  # Projection onto plane perpendicular to rotate_axis
    )

    node_features[:, :3] = torch.matmul(node_features[:, :3], rotate_mat)

    return node_features


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

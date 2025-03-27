"""Tensor-based augmentations of neuron structures for contrastive learning."""

import numpy as np
import torch
from torch.distributions.uniform import Uniform

from deep_neuronmorpho.utils.model_config import Augmentations


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
    jitter = torch.randn(3).numpy() * translate_var
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
            rotate_axis = torch.tensor([1, 0, 0], device=device)
        elif axis == "y":
            rotate_axis = torch.tensor([0, 1, 0], device=device)
        elif axis == "z":
            rotate_axis = torch.tensor([0, 0, 1], device=device)

    # Generate rotation angle
    angle_dist = Uniform(0, np.pi)
    theta = angle_dist.sample().to(device)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

    # Orthonormal unit vector along rotation axis
    u = rotate_axis / rotate_axis.norm()

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


def drop_branches(
    node_features: torch.Tensor,
    adjacency_list: torch.Tensor,
    n_branches: int,
    keep_nodes: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop a specified number of branches from the graph.

    This augmentation randomly selects branches to drop from the leaf nodes,
    and removes them along with any affected edges.

    Args:
        node_features: Tensor of shape (num_nodes, feat_dim) containing node features.
        adjacency_list: Tensor of shape (num_edges, 2) containing edge connections.
        n_branches: Number of branches to drop.
        keep_nodes: Minimum number of nodes to keep in graph.

    Returns:
        Tuple of (augmented node features, augmented adjacency list).
    """
    # TODO: Implement this
    raise NotImplementedError("Dropping branches is not implemented yet")


def augment_graph(
    node_features: torch.Tensor,
    adjacency_list: torch.Tensor,
    augmentations: Augmentations,
    keep_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a sequence of augmentations to node features and adjacency list.

    Args:
        node_features: Tensor of shape (num_nodes, feat_dim) containing node features.
        adjacency_list: Tensor of shape (num_edges, 2) containing edge connections.
        augmentations: Dictionary mapping augmentation types to their parameters.
                     The order of keys determines the sequence of application.

    Returns:
        Tuple of (augmented node features, augmented adjacency list).
    """

    if augmentations.num_drop_branches is not None:
        node_features, adjacency_list = drop_branches(
            node_features,
            adjacency_list,
            n_branches=augmentations.num_drop_branches,
            keep_nodes=keep_nodes,
        )

    if augmentations.jitter is not None:
        node_features = jitter_node_positions(node_features, jitter=augmentations.jitter)

    if augmentations.translate is not None:
        node_features = translate_all_nodes(node_features, translate_var=augmentations.translate)

    if augmentations.rotate_axis is not None:
        node_features = rotate_node_positions(node_features, axis=augmentations.rotate_axis)

    return node_features, adjacency_list

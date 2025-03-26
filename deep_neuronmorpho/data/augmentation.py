"""Tensor-based augmentations of neuron structures for contrastive learning."""

import numpy as np
import torch
from torch.distributions.uniform import Uniform


def perturb_node_positions(
    node_features: torch.Tensor,
    std_noise: float,
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
        std=std_noise,
        size=(node_features.shape[0], 3),
        device=node_features.device,
    )
    return node_features


def rotate_node_positions(node_features: torch.Tensor) -> torch.Tensor:
    """Perform a random 3D rotation on node coordinates.

    This augmentation generates a random rotation axis and random rotation angle,
    and creates a rotation matrix that rotates the input tensor along the given axis.

    Args:
        node_features: Tensor of shape (num_nodes, feat_dim) containing node features.
                       XYZ coordinates are assumed to be in the first 3 columns.

    Returns:
        Augmented node features tensor with rotated XYZ coordinates.
    """
    node_features = node_features.clone()
    device = node_features.device

    # Make sure we get a rotation axis
    rotate_axis = torch.tensor([0, 0, 0], device=device)
    while rotate_axis.sum() == 0:
        rotate_axis = torch.randint(2, (3,), device=device).float()

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
    prop: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop a proportion of nodes from the graph.

    This augmentation randomly selects nodes to drop based on a uniform distribution,
    and removes them along with any affected edges.

    Args:
        node_features: Tensor of shape (num_nodes, feat_dim) containing node features.
        adjacency_list: Tensor of shape (num_edges, 2) containing edge connections.
        prop: Proportion of nodes to drop.

    Returns:
        Tuple of (augmented node features, augmented adjacency list).
    """
    if prop == 0.0:
        return node_features, adjacency_list

    num_nodes = node_features.shape[0]

    # Generate random probability for each node based on position in the sequence
    # (Higher indices more likely to be dropped) - simple approximation without path distance
    node_indices = torch.arange(num_nodes, device=node_features.device).float()
    drop_probs = node_indices + 1  # Add 1 to avoid 0^power = 0 for node 0
    drop_probs[0] = 0.0  # Preserve the root node (index 0)
    # normalize to get probabilities
    drop_probs /= drop_probs.sum()
    # select nodes to drop
    drop_num = int(torch.ceil(num_nodes * torch.tensor(prop)).item())
    drop_nodes = torch.multinomial(drop_probs, min(drop_num, num_nodes - 1), replacement=False)
    # create a mask for nodes to keep
    keep_mask = torch.ones(num_nodes, dtype=torch.bool, device=node_features.device)
    keep_mask[drop_nodes] = False
    # create a mapping from old to new indices for the kept nodes
    idx_map = torch.zeros(num_nodes, dtype=torch.long, device=node_features.device)
    idx_map[keep_mask] = torch.arange(keep_mask.sum(), device=node_features.device)
    # filter edges to only include those between kept nodes
    edge_mask = keep_mask[adjacency_list[:, 0]] & keep_mask[adjacency_list[:, 1]]
    new_adjacency_list = idx_map[adjacency_list[edge_mask]]

    return node_features[keep_mask], new_adjacency_list


def augment_graph(
    node_features: torch.Tensor,
    adjacency_list: torch.Tensor,
    augmentations: dict[str, dict],
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
    for aug_type, params in augmentations.items():
        if aug_type == "rotate":
            node_features = rotate_node_positions(node_features)
        elif aug_type == "perturb":
            node_features = perturb_node_positions(node_features, std_noise=params["jitter"])
        elif aug_type == "drop_branches":
            node_features, adjacency_list = drop_branches(
                node_features,
                adjacency_list,
                prop=params["prop"],
            )

    return node_features, adjacency_list

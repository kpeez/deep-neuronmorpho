"""Graph augmentations of neuron structures for contrastive learning."""
import numpy as np
import pandas as pd
import torch
from torch.distributions.uniform import Uniform


def perturb_points(
    node_features: torch.Tensor, prop: float = 0.5, std_noise: float = 2.0
) -> torch.Tensor:
    """Shift a proportion of points by a random distance with Gaussian noise.

    Assumes xyz coordinates are in the first 3 columns of node_features.
    Each point's 3D coordinates are updated:

    x' = x + dx, y' = y + dy, z' = z + dz

    Args:
        node_features (torch.Tensor): Node features. First 3 columns are assumed to be xyz
        coordinates.
        prop (float, optional): Proportion of points to perturb. Defaults to 0.5.
        std_noise (float, optional): Standard deviation of Gaussian noise. Defaults to 2.0.

    Returns:
        torch.Tensor: Node features with perturbed points.
    """
    num_nodes = node_features.shape[0]
    nodes_to_perturb = np.random.choice(
        num_nodes,
        int(num_nodes * prop),
        replace=False,
    )
    node_features[nodes_to_perturb, :3] += torch.normal(
        mean=0,
        std=std_noise,
        size=(len(nodes_to_perturb), 3),
        device=node_features.device,
    )

    return node_features


def graph_rotation(node_features: torch.Tensor) -> torch.Tensor:
    """Perform a rotation operation on a node features tensor for a single graph.

    This function generates a random rotation axis and random rotation angle, and uses these to
    create a rotation matrix that rotates the input tensor along the given axis by the given angle.

    Args:
        node_features (torch.Tensor): Input tensor of shape (batch_size, num_nodes, num_features).

    Returns:
        torch.Tensor: Rotated node features tensor of the same shape as the input.
    """
    device = node_features.device
    # Make sure we get a rotation axis
    rotate_axis = torch.tensor([0, 0, 0], device=device)
    while rotate_axis.sum() == 0:
        rotate_axis = torch.randint(2, (3,), device=device).float()

    # Generate rotation angle
    angle_dist = Uniform(0, np.pi)
    theta = torch.tensor(angle_dist.sample().item() * (np.pi / 180.0))
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    # orthonormal unit vector along rotation axis
    u = rotate_axis / rotate_axis.norm()
    # outer product of u with itself used to project vectors onto the plane perpendicular to u.
    outer = torch.ger(u, u)
    # this matrix rotates vectors along `u` axis by angle `theta`
    rotate_mat = (
        cos_theta * torch.eye(3, device=device)  # rotation about rotate_axis
        + sin_theta
        * torch.tensor(  # rotation about plane perpendicular to rotate_axis
            [
                [0, -u[2], u[1]],
                [u[2], 0, -u[0]],
                [-u[1], u[0], 0],
            ],
            device=device,
        )
        + (1 - cos_theta) * outer  # projection onto plane perpendicular to rotate_axis
    )
    node_features[:, :3] = torch.matmul(node_features[:, :3], rotate_mat)

    return node_features


def drop_branches(neuron_swc: pd.DataFrame) -> pd.DataFrame:
    """Drop a random branch from a neuron."""
    # TODO:
    # 1. implement probability of branch selection from ZhaoEtAl2022
    # 2. select 2% of branches to drop based on probability from 1.
    # 3. drop branches from neuron

    raise NotImplementedError

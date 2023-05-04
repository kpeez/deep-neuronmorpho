"""Graph augmentations of neuron structures for contrastive learning."""
from copy import deepcopy
from typing import Callable

import dgl
import numpy as np
import torch
from dgl import DGLGraph
from torch.distributions.uniform import Uniform


class GraphAugmenter:
    """A class to perform graph augmentations on DGLGraphs.

    Currently supported augmentations are:
        - Perturb node positions.
            - args taken: prop (proportion of points to perturb), std_noise (std of Gaussian noise)
        - Rotate node positions
            - args taken: None (random rotation angle and axis are generated)

    Args:
        augmentation_params (dict): A dictionary with augmentation types as keys and their
        respective parameters as values. Supported augmentation types are 'perturb' and 'rotate'.

    Methods:
        augment_batch: Augments a batch of DGLGraphs using the specified augmentation parameters.
        drop_branches: Not yet implemented.
        perturb_node_positions (staticmethod): Shifts a proportion of points by a random distance.
        rotate_graph_nodes (staticmethod): Performs a rotation on xyz node features for a graph.

    Example:
        augmentations = {"perturb": {"prop": 0.5, "std_noise": 2.0}, "rotate": {}}
        augmenter = GraphAugmenter(augmentations)
        augmented_graphs = augmenter.augment_batch(g_batch)
    """

    def __init__(self, augmentation_params: dict[str, dict[str, str | float]] | None = None):
        self.augmentation_params = augmentation_params or {}
        self.augmentation_funcs: dict[str, Callable[..., DGLGraph]] = {
            "perturb": self.perturb_node_positions,
            "rotate": self.rotate_graph_nodes,
        }
        self.validate_augmentations()

        self.augmented_graphs = None

    def validate_augmentations(self) -> None:
        """Validate the augmentation parameters."""
        for aug in self.augmentation_params:
            if aug not in self.augmentation_funcs:
                raise ValueError(f"Augmentation function: '{aug}' not supported.")

    @staticmethod
    def perturb_node_positions(
        graph: DGLGraph,
        prop: float = 0.5,
        std_noise: float = 2.0,
    ) -> DGLGraph:
        """Shift a proportion of points by a random distance with Gaussian noise.

        Node features are assumed to be in the graph.ndata["nattrs"] tensor.
        The xyz coordinates are expected to be in the first 3 columns of node_features.
        Each point's 3D coordinates are updated:

        x' = x + dx, y' = y + dy, z' = z + dz

        Args:
            graph (DGLGraph): DGLGraph with node features accessible via graph.ndata["nattrs"].
            First 3 columns are assumed to be xyz coordinates.
            prop (float, optional): Proportion of points to perturb. Defaults to 0.5.
            std_noise (float, optional): Standard deviation of Gaussian noise. Defaults to 2.0.

        Returns:
            DGLGraph with perturbed node features.
        """
        new_graph = deepcopy(graph)
        node_features = new_graph.ndata["nattrs"]
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
        new_graph.ndata["nattrs"] = node_features

        return new_graph

    @staticmethod
    def rotate_graph_nodes(graph: DGLGraph) -> DGLGraph:
        """Perform a rotation operation on xyz node features for a graph.

        Node features are assumed to be in the graph.ndata["nattrs"] tensor.

        This function generates a random rotation axis and random rotation angle, and uses these to
        create a rotation matrix to rotate the input tensor along the given axis by the given angle.

        Args:
            graph (DGLGraph): Input tensor of shape (batch_size, num_nodes, num_features).

        Returns:
            DGLGraph with rotated node positions.
        """
        new_graph = deepcopy(graph)
        node_features = new_graph.ndata["nattrs"]
        device = node_features.device
        # Make sure we get a rotation axis
        rotate_axis = torch.tensor([0, 0, 0], device=device)
        while rotate_axis.sum() == 0:
            rotate_axis = torch.randint(2, (3,), device=device).float()
        # Generate rotation angle
        angle_dist = Uniform(0, np.pi)
        theta = angle_dist.sample().to(device)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
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
        new_graph.ndata["nattrs"] = node_features

        return new_graph

    @staticmethod
    def drop_branches(g_batch: DGLGraph) -> DGLGraph:
        # TODO: Implement random branch dropping
        raise NotImplementedError("This method is not yet implemented.")

    def augment_batch(self, g_batch: DGLGraph) -> DGLGraph:
        """Augments a batch of DGLGraphs using the specified augmentation parameters.

        Args:
            g_batch (DGLGraph): A batch of DGLGraphs to be augmented.

        Returns:
            DGLGraph: A batch of augmented DGLGraphs.
        """
        original_graphs = dgl.unbatch(g_batch)
        augmented_graphs = []
        for g in original_graphs:
            aug_g = g
            for aug_type, aug_params in self.augmentation_params.items():
                aug_g = self.augmentation_funcs[aug_type](aug_g, **aug_params)
            augmented_graphs.append(aug_g)

        batch_augmented_graphs = dgl.batch(augmented_graphs)
        self.augmented_graphs = batch_augmented_graphs

        return self.augmented_graphs

    def __repr__(self) -> str:
        """Return representation of a GraphAugmenter object."""
        return f"GraphAugmenter(augmentation params:{self.augmentation_params})"

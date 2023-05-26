"""Graph augmentations of neuron structures for contrastive learning."""
import inspect
from abc import ABC, abstractmethod
from copy import deepcopy

import dgl
import numpy as np
import torch
from dgl import DGLGraph
from torch.distributions.uniform import Uniform

from deep_neuronmorpho.utils import ModelConfig


class GraphAugmentation(ABC):
    """An abstract base class for defining graph augmentation operations.

    This class serves as a blueprint for creating custom graph augmentation classes.
    Each subclass should implement the `apply` method to perform the desired augmentation
    on a given DGLGraph.

    Methods:
        apply: Applies the augmentation to the input graph. This method must be implemented
            by the subclasses.

    Usage:
    ```
        class CustomGraphAugmentation(GraphAugmentation):
            def __init__(self, param1, param2):
                self.param1 = param1
                self.param2 = param2

            def apply(self, graph: DGLGraph) -> DGLGraph:
                # Perform the desired augmentation using self.param1 and self.param2
                # ...

        custom_augmentation = CustomGraphAugmentation(param1, param2)
    ```
    """

    @abstractmethod
    def apply(self, graph: DGLGraph) -> DGLGraph:
        """Apply the augmentation to the input graph."""
        pass

    def __repr__(self) -> str:
        """Return a string representation of the augmentation."""
        init_signature = inspect.signature(self.__class__.__init__)
        params = init_signature.parameters
        params_repr = ", ".join(
            f"{name}={getattr(self, name)}" for name in params if name != "self"
        )
        return f"{self.__class__.__name__}({params_repr})"


class PerturbNodePositions(GraphAugmentation):
    """Shift a proportion of points by a random distance with Gaussian noise.

    This augmentation assumes that node features are stored in the graph.ndata["nattrs"] tensor,
    with the xyz coordinates in the first three columns of the node features.
    Each point's 3D coordinates are updated as follows:

    x' = x + dx, y' = y + dy, z' = z + dz

    Args:
        prop (float): Proportion of points to perturb.
        std_noise (float): Standard deviation of Gaussian noise.

    Methods:
        apply: Apply the perturbation to the input DGLGraph's node features.

    Example:
        perturb_aug = PerturbNodePositions(prop=0.5, std_noise=2.0)
        perturbed_graph = perturb_aug.apply(graph)
    """

    def __init__(self, prop: float, std_noise: float):
        self.prop = prop
        self.std_noise = std_noise

    def apply(self, graph: DGLGraph) -> DGLGraph:
        """Apply the augmentation to the input graph."""
        node_features = graph.ndata["nattrs"]
        num_nodes = node_features.shape[0]
        nodes_to_perturb = np.random.choice(
            num_nodes,
            int(num_nodes * self.prop),
            replace=False,
        )
        node_features[nodes_to_perturb, :3] += torch.normal(
            mean=0,
            std=self.std_noise,
            size=(len(nodes_to_perturb), 3),
            device=node_features.device,
        )
        graph.ndata["nattrs"] = node_features

        return graph


class RotateGraphNodes(GraphAugmentation):
    """Perform a rotation operation on xyz node features for a graph.

    This augmentation assumes that node features are stored in the graph.ndata["nattrs"] tensor.
    The augmentation generates a random rotation axis and random rotation angle, and uses these
    to create a rotation matrix that rotates the input tensor along the given axis.

    Methods:
        apply: Apply the rotation to the input DGLGraph's node features.

    Example:
        rotate_aug = RotateGraphNodes()
        rotated_graph = rotate_aug.apply(graph)
    """

    def __init__(self) -> None:
        pass

    def apply(self, graph: DGLGraph) -> DGLGraph:
        """Apply the augmentation to the input graph."""
        node_features = graph.ndata["nattrs"]
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
        graph.ndata["nattrs"] = node_features

        return graph


class DropBranches(GraphAugmentation):
    """Randomly drop a proportion of nodes along with their descendants in the graph.

    This augmentation drops nodes based on their path distance from the root (soma), treating nodes
    with larger distances as more likely to be dropped. The path distance of a node is assumed to be
    stored in the graph.ndata["nattrs"] tensor, specified by the path_dist_idx parameter.

    Args:
        prop (float): Proportion of nodes to drop.
        deg_power (float): Power to which to raise each node's path distance when computing P(drop).
        path_dist_idx (int): Column index in the node features tensor for path distance from root.

    Methods:
        apply: Apply augmentation to the input DGLGraph by randomly dropping nodes (and children).

    Example:
        drop_branches_aug = DropBranches(prop=0.25, deg_power=1.0, path_dist_idx=3)
        pruned_graph = drop_branches_aug.apply(graph)

    """

    def __init__(self, prop: float, deg_power: float = 1.0, path_dist_idx: int = 3) -> None:
        self.prop = prop
        self.deg_power = deg_power
        self.path_dist_idx = path_dist_idx

    def get_descendants(self, graph: DGLGraph, node: int, path_dist: list[float]) -> list[int]:
        """Recursively find and return all descendants of a given node in the graph."""
        descendants = []
        for child in graph.successors(node):
            if path_dist[child] > path_dist[node]:
                descendants.append(child.item())
                descendants.extend(self.get_descendants(graph, child, path_dist))
        return descendants

    def apply(self, graph: DGLGraph) -> DGLGraph:
        """Apply the augmentation to the input graph."""
        path_dist = graph.ndata["nattrs"][:, self.path_dist_idx]
        # Compute selection probabilities
        path_dist_p = (path_dist + torch.abs(torch.min(path_dist))) ** self.deg_power  # keep soma
        select_probs = path_dist_p / torch.sum(path_dist_p)
        # Select nodes to drop
        drop_num = int(torch.ceil(graph.num_nodes() * torch.tensor(self.prop)).item())
        drop_nodes_tensor = torch.multinomial(select_probs, drop_num, replacement=False)
        # Add descendants of selected nodes
        drop_nodes = set(drop_nodes_tensor.tolist())
        for node in list(drop_nodes):
            drop_nodes.update(self.get_descendants(graph, node, path_dist))
        # Remove the dropped nodes
        remaining_nodes = list(set(range(graph.number_of_nodes())) - drop_nodes)
        subgraph = graph.subgraph(remaining_nodes)

        return subgraph


class GraphAugmenter:
    """A class to perform graph augmentations on DGLGraphs using a modular approach.

    This class supports the following augmentations:
        - Perturb node positions
            - args taken: prop (proportion of points to perturb), std_noise (std of Gaussian noise)
        - Rotate node positions
            - args taken: None (random rotation angle and axis are generated)
        - Drop branches
            - args taken: prop (proportion of nodes to drop), deg_power (power for path distance),
            path_dist_idx (path distance index in node features)

    Methods:
        augment_batch: Augments a batch of DGLGraphs using the specified augmentations.
    ```
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.augmentations = self.parse_config()

    def parse_config(self) -> list[GraphAugmentation]:
        """Parse the augmentation config and return a list of GraphAugmentation objects."""
        aug_order = self.config.order
        aug_params = self.config.params.to_dict()

        augmentation_classes = {
            "perturb": PerturbNodePositions,
            "rotate": RotateGraphNodes,
            "drop_branches": DropBranches,
        }

        augmentations = []
        for aug_type in aug_order:
            if aug_type in augmentation_classes:
                assert isinstance(aug_params, dict)
                params = aug_params.get(aug_type, {})
                augmentations.append(augmentation_classes[aug_type](**params))
            else:
                raise ValueError(f"Unknown augmentation type: {aug_type}")

        return augmentations

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
            aug_g = deepcopy(g)
            for augmentation in self.augmentations:
                aug_g = augmentation.apply(aug_g)
            augmented_graphs.append(aug_g)
        batch_augmented_graphs = dgl.batch(augmented_graphs)

        return batch_augmented_graphs

    def __repr__(self) -> str:
        """Return a string representation of the augmentation."""
        augmentations_repr = ", ".join(repr(augmentation) for augmentation in self.augmentations)
        return f"{self.__class__.__name__}({augmentations_repr})"

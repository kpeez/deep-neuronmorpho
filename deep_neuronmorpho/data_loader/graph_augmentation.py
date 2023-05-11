"""Graph augmentations of neuron structures for contrastive learning."""
import inspect
from abc import ABC, abstractmethod
from copy import deepcopy

import dgl
import numpy as np
import torch
from dgl import DGLGraph
from torch.distributions.uniform import Uniform


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
    # TODO: Implement random branch dropping
    pass


class GraphAugmenter:
    """A class to perform graph augmentations on DGLGraphs using a modular approach.

    This class supports the following augmentations:
        - Perturb node positions
            - args taken: prop (proportion of points to perturb), std_noise (std of Gaussian noise)
        - Rotate node positions
            - args taken: None (random rotation angle and axis are generated)
        - Drop branches (not yet implemented)

    Args:
        config (dict[str, list[str] | dict[str, dict[str, float]]]): A dictionary containing the
            augmentation order and parameters. The dictionary should have the following structure:
            {
                'order': [augmentation_type1, augmentation_type2, ...],
                'params': {
                    augmentation_type1: {param1: value1, param2: value2, ...},
                    augmentation_type2: {param1: value1, param2: value2, ...},
                    ...
                }
            }
            Supported augmentation types are 'perturb', 'rotate', and 'drop_branches'.

    Attributes:
        config (dict[str, list[str] | dict[str, dict[str, float]]]): The provided configuration
            dictionary.
        augmentations (list[GraphAugmentation]): A list of GraphAugmentation objects created based
            on the provided configuration.

    Methods:
        parse_config: Parses the configuration dictionary and returns a list of GraphAugmentation
            objects.
        augment_batch: Augments a batch of DGLGraphs using the specified augmentations.

    Example:
        config = {
            'order': ['perturb', 'rotate'],
            'params': {
                'perturb': {'prop': 0.5, 'std_noise': 2.0},
                'rotate': {}
            }
        }
        augmenter = GraphAugmenter(config)
        augmented_graphs = augmenter.augment_batch(g_batch)
    """

    def __init__(self, config: dict[str, list[str] | dict[str, dict[str, float]]]):
        self.config = config
        self.augmentations = self.parse_config()

    def parse_config(self) -> list[GraphAugmentation]:
        """Parse the augmentation config and return a list of GraphAugmentation objects."""
        aug_order = self.config["order"]
        aug_params = self.config["params"]

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

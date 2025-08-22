"""Augmentations and transformations for contrastive learning."""

from typing import Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .graph_features import compute_neuron_node_feats


class RandomTranslate(BaseTransform):
    """
    Translates entire graph by the same random offset.
    This is what you want for "translate nodes" - moving the whole graph together.
    """

    def __init__(self, translate: float | Sequence[float]):
        """
        Args:
            translate (float or sequence): Maximum translation in each dimension.
                If float, same value is used for all dimensions.
                If sequence, should match the number of dimensions in pos.
        """
        if isinstance(translate, (int, float)):
            self.translate = translate
        else:
            self.translate = translate

    def __call__(self, data):
        if not hasattr(data, "pos") or data.pos is None:
            return data

        pos = data.pos
        dim = pos.size(-1)

        if isinstance(self.translate, (int, float)):
            translation = torch.uniform(-self.translate, self.translate, (dim,))
        else:
            translation = torch.tensor(
                [torch.uniform(-t, t, (1,)).item() for t in self.translate[:dim]]
            )
        data.pos += translation.unsqueeze(0)
        return data


class DropRandomBranches(BaseTransform):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, data: Data) -> Data:
        raise NotImplementedError


class RecomputeNodeFeatures(BaseTransform):
    """
    Recomputes the node features using the current data.pos values.
    Use at the end of your transform chain to ensure .x always reflects the transformed positions.

    Node features: \n
        1. x: x coordinate of node.
        2. y: y coordinate of node.
        3. z: z coordinate of node.
        4. radial_log: log of radial distance from soma.
        5. path_log: log of path distance from soma.
        6. tortuosity: tortuosity of the path.
        7. branch_order: branch order of the node.
        8. strahler_order: strahler order of the node.

    See `compute_neuron_node_feats` for more details.
    """

    def __call__(self, data: Data) -> Data:
        root = int(getattr(data, "root", 0))
        data.x = compute_neuron_node_feats(data.pos, data.edge_index, root)
        return data

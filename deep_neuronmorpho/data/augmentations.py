"""Augmentations and transformations for contrastive learning."""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomTranslate(BaseTransform):
    """
    Translates entire graph by the same random offset.
    This is what you want for "translate nodes" - moving the whole graph together.
    """

    def __init__(self, translate):
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


class DropRandomBranch(BaseTransform):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, data: Data) -> Data:
        raise NotImplementedError


class SyncPosToFeatures(BaseTransform):
    """
    Updates the first 3 dimensions of data.x with the current data.pos values.
    Use at the end of your transform chain to ensure .x always reflects the transformed positions.
    """

    def __init__(self, pos_dims=3):
        """Number of position dimensions to sync (3 for x,y,z)."""
        self.pos_dims = pos_dims

    def __call__(self, data):
        if (
            hasattr(data, "pos")
            and data.pos is not None
            and hasattr(data, "x")
            and data.x is not None
        ):
            actual_pos_dims = min(self.pos_dims, data.pos.size(-1), data.x.size(-1))
            data.x[:, :actual_pos_dims] = data.pos[:, :actual_pos_dims]

        return data

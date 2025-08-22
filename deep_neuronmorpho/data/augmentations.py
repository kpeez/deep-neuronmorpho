"""Augmentations and transformations for contrastive learning."""

import math
from typing import Sequence

import torch
from torch import Tensor
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(translate={self.translate})"


class DropRandomBranches(BaseTransform):
    """
    Randomly drops entire branches (subtrees) from a neuron graph.

    This transform serves as a structural data augmentation, creating realistic
    variations of a neuron's morphology. It ensures that no isolated nodes are
    created by always removing complete subtrees.

    - The selection algorithm uses a vectorized, weighted random sampling strategy (weighted by subtree size).
    - The process selects branches that fit within the budget and makes a final "best-effort" choice
    to land as close to the target as possible if no perfect fits remain.
    - The selection can be biased using the `alpha` parameter. Setting `alpha > 1` will favor
      dropping larger branches, while `alpha < 1` will favor smaller ones.

    Args:
        drop_fraction (float): Target fraction of nodes to drop (0.0-0.3 recommended).
        min_keep_nodes (int): Minimum number of nodes to retain in the graph.
        alpha (float): Weight ∝ size**alpha.
        recompute_features (bool): Whether to recompute the node features after dropping branches.

    Returns:
        New Data object with dropped branches and recomputed features.
    """

    def __init__(
        self,
        drop_fraction: float = 0.15,
        min_keep_nodes: int = 50,
        alpha: float = 1.0,
        recompute_features: bool = False,
    ):
        if drop_fraction < 0 or drop_fraction > 1:
            raise ValueError(f"drop_fraction must be in [0, 1], got {drop_fraction}")
        if min_keep_nodes < 1:
            raise ValueError(f"min_keep_nodes must be >= 1, got {min_keep_nodes}")

        self.drop_fraction = drop_fraction
        self.min_keep_nodes = min_keep_nodes
        self.alpha = alpha
        self.recompute_features = recompute_features

    def __call__(self, data: Data) -> Data:
        """
        Apply branch dropping to the input graph.

        Args:
            data: PyG Data object with required attributes:
                - pos: [N, 3] node positions
                - edge_index: [2, E] edge connections
                - dfs_entry: [N] DFS entry indices
                - dfs_exit: [N] DFS exit indices
                - root: int (root node index, typically 0)

        Returns:
            New Data object with dropped branches and recomputed features.
        """
        required_attrs = ["pos", "edge_index", "dfs_entry", "dfs_exit", "root"]
        for attr in required_attrs:
            if not hasattr(data, attr):
                raise ValueError(f"Data object missing required attribute: {attr}")

        N = data.pos.size(0)
        device = data.pos.device
        drop_mask = self._select_branches_to_drop(
            dfs_entry=data.dfs_entry,
            dfs_exit=data.dfs_exit,
            N=N,
            root=data.root,
            drop_fraction=self.drop_fraction,
            min_keep_nodes=self.min_keep_nodes,
            device=device,
            alpha=self.alpha,
        )
        keep_mask = ~drop_mask
        new_pos, new_edge_index = self._filter_graph_structure(
            pos=data.pos,
            edge_index=data.edge_index,
            keep_mask=keep_mask,
        )
        new_data = Data(pos=new_pos, edge_index=new_edge_index, root=0)
        for key, value in data:
            if key in {"pos", "edge_index", "x", "dfs_entry", "dfs_exit", "root"}:
                continue
            if torch.is_tensor(value) and value.dim() > 0 and int(value.size(0)) == N:
                new_data[key] = value[keep_mask]
            else:
                new_data[key] = value

        if self.recompute_features:
            new_data.x = compute_neuron_node_feats(new_data.pos, new_data.edge_index, new_data.root)

        return new_data

    def _select_branches_to_drop(
        self,
        dfs_entry: Tensor,
        dfs_exit: Tensor,
        N: int,
        root: int,
        drop_fraction: float,
        min_keep_nodes: int,
        device: torch.device,
        alpha: float = 1.0,
        extra_weights: Tensor | None = None,
    ) -> Tensor:
        """
        Selects which branches to drop based on DFS entry/exit indices.

        Args:
            dfs_entry: [N] preorder entry indices
            dfs_exit: [N] preorder exit indices
            N: number of nodes
            root: root node index
            drop_fraction: target fraction of nodes to drop
            min_keep_nodes: minimum number of nodes to keep
            device: device to use
            alpha: weight ∝ size**alpha

        Returns:
            [N] bool mask over NODE IDs (True = drop)
        """
        K = max(int(min_keep_nodes), math.ceil((1.0 - float(drop_fraction)) * N))
        K = max(1, min(K, N - 1))
        # sizes & weights
        sizes = (dfs_exit - dfs_entry + 1).to(torch.float32).clamp_min(1)
        w = sizes ** float(alpha)
        if extra_weights is not None:
            w = w * extra_weights.to(w)
        # candidates: exclude root
        cand = torch.arange(N - 1, device=device) + 1
        # size-weighted random order (Gumbel/Exp trick)
        keys = -torch.log(torch.rand_like(w[cand])) / w[cand].clamp_min(1)
        order = cand[torch.argsort(keys)]  # small key first
        # preorder coverage & node-wise mask
        covered_pre = torch.zeros(N, dtype=torch.bool, device=device)
        drop_nodes = torch.zeros(N, dtype=torch.bool, device=device)
        # map: preorder index -> node id
        pre2node = torch.empty(N, dtype=torch.long, device=device)
        pre2node[dfs_entry] = torch.arange(N, device=device)

        remaining_keep = N
        best_over = None
        best_gap = float("inf")  # how far below K we'd land

        for i in order.tolist():
            s = int(dfs_entry[i].item())
            t = int(dfs_exit[i].item())
            if covered_pre[s]:
                continue
            sz = t - s + 1

            # take if we can stay >= K keeps
            if remaining_keep - sz >= K:
                covered_pre[s : t + 1] = True
                drop_nodes[pre2node[s : t + 1]] = True
                remaining_keep -= sz
            else:
                # remember best feasible overshoot (stay >= min_keep_nodes)
                post = remaining_keep - sz
                if post >= min_keep_nodes:
                    gap = K - post  # >= 0
                    if gap < best_gap:
                        best_gap = gap
                        best_over = (s, t, sz)

        # one-shot overshoot if still above K
        if remaining_keep > K and best_over is not None:
            s, t, sz = best_over
            covered_pre[s : t + 1] = True
            drop_nodes[pre2node[s : t + 1]] = True

        drop_nodes[int(root)] = False

        return drop_nodes

    def _filter_graph_structure(
        self, pos: Tensor, edge_index: Tensor, keep_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Filter positions and edges based on keep_mask."""
        N = pos.size(0)
        device = pos.device
        keep_indices = torch.where(keep_mask)[0]
        # map old -> new indices
        new_idx = torch.full((N,), -1, dtype=torch.long, device=device)
        new_idx[keep_indices] = torch.arange(len(keep_indices), device=device)
        # filter and remap edges
        new_pos = pos[keep_mask]
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        new_edge_index = new_idx[edge_index[:, edge_mask]]

        return new_pos, new_edge_index

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"drop_fraction={self.drop_fraction}, "
            f"min_keep_nodes={self.min_keep_nodes}, "
            f"recompute_features={self.recompute_features})"
        )


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

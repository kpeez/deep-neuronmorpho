"""Graph dataset class from ssl_neuron.

TODO: This is a temporary class to load the dataset from ssl_neuron. Will need to rewrite
to utilize torch-geometric's dataset class.
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from ..data.utils import (
    compute_path_lengths,
    drop_random_branch,
    find_leaf_nodes,
    neighbors_to_adjacency_torch,
    remap_neighbors,
    subsample_graph,
)
from .augmentations import (
    jitter_node_positions,
    rotate_node_positions,
    translate_all_nodes,
)


class GraphDINODataset(Dataset):
    """Dataset of neuronal graphs for training GraphDINO.

    Neuronal graphs are assumed to be soma-centered (i.e. soma node
    position is (0, 0, 0) and axons have been removed. Node positions
    are assumed to be in microns and y-axis is orthogonal to the pia.
    """

    def __init__(self, cfg: DictConfig, mode="train"):
        self.cfg = cfg
        self.mode = mode
        dataset_folder = cfg.data.train_dataset if self.mode == "train" else cfg.data.eval_dataset

        if dataset_folder is None:
            raise ValueError(f"Dataset for mode '{self.mode}' is not specified in the config.")

        data_path = Path(dataset_folder)
        data_path = data_path.resolve()
        self.cell_files = sorted(data_path.glob("*.pkl"))
        self.num_samples = len(self.cell_files)
        if self.num_samples == 0:
            raise ValueError(f"No graph files found in {data_path}")

        # augmentation parameters
        self.jitter_var = cfg.augmentations.jitter
        self.rotation_axis = cfg.augmentations.rotation_axis
        self.n_drop_branch = cfg.augmentations.num_drop_branches
        self.translate_var = cfg.augmentations.translate
        self.n_nodes = cfg.data.num_nodes

    def __len__(self):
        return self.num_samples

    def _delete_subbranch(self, neighbors, soma_id, distances, leaf_branch_nodes):
        leaf_branch_nodes = set(leaf_branch_nodes)
        not_deleted = set(range(len(neighbors)))
        for _i in range(self.n_drop_branch):
            neighbors, drop_nodes = drop_random_branch(
                leaf_branch_nodes, neighbors, distances, keep_nodes=self.n_nodes
            )
            not_deleted -= drop_nodes
            leaf_branch_nodes -= drop_nodes

            if len(leaf_branch_nodes) == 0:
                break

        return not_deleted

    def _reduce_nodes(self, neighbors, soma_id, distances, leaf_branch_nodes):
        """Reduce the number of nodes in the graph to a fixed size.

        This method performs two types of reduction:
        1. Delete random branches using _delete_subbranch
        2. Subsample the graph to a fixed number of nodes using subsample_graph

        Args:
            neighbors: Dict mapping node IDs to their connected neighbors
            soma_id: ID of the soma node(s) to protect from deletion
            distances: Dict mapping node IDs to their distance from soma
            leaf_branch_nodes: Set of leaf and branch nodes

        Returns:
            neighbors2: Updated neighbors dictionary
            adj_matrix: Adjacency matrix for GraphDINO
            not_deleted: List of node IDs that were not deleted
        """
        neighbors2 = {k: set(v) for k, v in neighbors.items()}
        not_deleted = self._delete_subbranch(neighbors2, soma_id, distances, leaf_branch_nodes)
        neighbors2, not_deleted = subsample_graph(
            neighbors=neighbors2,
            not_deleted=not_deleted,
            keep_nodes=self.n_nodes,
            protected=soma_id,
        )

        adj_matrix = neighbors_to_adjacency_torch(neighbors2, not_deleted)
        assert adj_matrix.shape == (self.n_nodes, self.n_nodes), (
            f"{adj_matrix.shape} != {self.n_nodes} {self.n_nodes}"
        )

        return neighbors2, adj_matrix, not_deleted

    def _augment_node_position(self, features):
        pos = features[:, :3]
        rot_pos = rotate_node_positions(pos, axis=self.rotation_axis)
        jittered_pos = jitter_node_positions(rot_pos, jitter=self.jitter_var)
        jittered_pos = translate_all_nodes(
            jittered_pos,
            translate_var=self.cfg.augmentations.translate,
        )
        features[:, :3] = jittered_pos

        return features

    def _augment(self, cell):
        features = cell["features"]
        neighbors = cell["neighbors"]
        distances = cell["distances"]
        _neighbors2, adj_matrix, not_deleted = self._reduce_nodes(
            neighbors, [int(cell["soma_id"])], distances, cell["leaf_branch_nodes"]
        )

        new_features = features[not_deleted].clone()
        new_features = self._augment_node_position(new_features)

        return new_features, adj_matrix

    def __getitem__(self, index):
        cell_file = self.cell_files[index]
        soma_id = 0
        cell_data = np.load(cell_file, allow_pickle=True)
        features = cell_data["features"]
        neighbors = cell_data["neighbors"]

        assert len(features) == len(neighbors)

        if len(features) >= self.n_nodes or self.mode == "eval":
            neighbors, _ = subsample_graph(
                neighbors=neighbors,
                not_deleted=set(range(len(neighbors))),
                keep_nodes=1000,
                protected=[soma_id],
            )
            neighbors, subsampled2new = remap_neighbors(neighbors)
            soma_id = subsampled2new[soma_id]
            features = features[list(subsampled2new.keys()), :3]

            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)

            leaf_branch_nodes = find_leaf_nodes(neighbors)
            distances = compute_path_lengths(soma_id, neighbors)

            cell = {
                "cell_id": cell_file.stem,
                "features": features,
                "neighbors": neighbors,
                "distances": distances,
                "soma_id": soma_id,
                "leaf_branch_nodes": leaf_branch_nodes,
            }

            if self.mode == "train":
                features1, adj_matrix1 = self._augment(cell)
                features2, adj_matrix2 = self._augment(cell)
                return features1, features2, adj_matrix1, adj_matrix2
            else:
                return cell["features"], cell["neighbors"]
        else:
            # we should probably skip graphs that are smaller than n_nodes when not in eval mode
            # for now, we'll return None and it should be handled in the dataloader collate_fn
            return None

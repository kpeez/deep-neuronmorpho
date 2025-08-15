"""
Tests for building a PyG Data from an SWC DataFrame.
"""

import numpy as np
import pandas as pd
import torch

from deep_neuronmorpho.data import swc_df_to_pyg_data


def test_synthetic_swc_dfs_invariants(synthetic_swc_dataframe: pd.DataFrame):
    # Simple tree:
    # 1 (-1)
    # ├─ 2 (1)
    # └─ 3 (1)
    #     └─ 4 (3)
    data = swc_df_to_pyg_data(synthetic_swc_dataframe)
    # N = 4 nodes; E = 3 directed edges
    assert data.x.shape == (4, 3)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == 3
    # check root is first in DFS order (id = 1)
    assert data.root == 0
    assert int(data.orig_id[0]) == 1
    # check positions alias
    assert torch.equal(data.pos, data.x)
    # traversal indices attached by default
    assert hasattr(data, "dfs_entry") and hasattr(data, "dfs_exit")
    assert data.dfs_entry.shape == (4,)
    assert data.dfs_exit.shape == (4,)
    assert torch.all(data.dfs_entry <= data.dfs_exit)


def test_real_swc_file_order_roundtrip_if_available(swc_dataframe: pd.DataFrame):
    data = swc_df_to_pyg_data(swc_dataframe)
    # reconstruct positions in original file order using orig_id mapping
    pos_file = torch.tensor(swc_dataframe[["x", "y", "z"]].to_numpy(), dtype=torch.float32)
    id2row = {int(n): i for i, n in enumerate(swc_dataframe["n"].to_numpy(np.int64))}
    perm = torch.tensor([id2row[int(i)] for i in data.orig_id.tolist()], dtype=torch.long)
    pos_new_in_file_order = data.x[perm]

    torch.testing.assert_close(pos_new_in_file_order, pos_file, rtol=1e-6, atol=1e-6)
